# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import datetime
import json
import logging
import os
from os import system

import visdom
from dateutil import tz
from sacred.observers.base import RunObserver

from src.utils.plotting import get_env_name

logger = logging.getLogger(__name__)


def utc_to_local(datetime):
    utc = datetime.replace(tzinfo=tz.tzutc())
    return utc.astimezone(tz.tzlocal())

def flatten_one_level(d):
    new_d = {}
    for out_k, out_v in d.items():
        if not isinstance(out_v, collections.Mapping):
            new_d[out_k] = out_v
            continue

        new_key_f = '{}.{}'
        for in_k, in_v in out_v.items():
            new_d[new_key_f.format(out_k, in_k)] = in_v
    return new_d


class LogObserver(RunObserver):

    @staticmethod
    def create(visdom_opts, *args, **kwargs):
        return LogObserver(visdom_opts, *args, **kwargs)

    def __init__(self, visdom_opts, *args, **kwargs):
        super(LogObserver, self).__init__(*args, **kwargs)
        self.visdom_opts = visdom_opts
        self.viz = None
        self.config = None
        self.run_id = None
        self.exp_name = None
        self.start_time = None

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        self.config = config
        self.run_id = _id
        self.exp_name = get_env_name(config, _id, main=True)
        self.start_time = start_time

        # Ugly trick to set the pane name if using GNU Screen
        system("echo '\ek{}\e\\'".format(self.run_id))

        traces_folder = config['experiment']['visdom_traces_folder']
        self.traces_folder = os.path.join(traces_folder,
                                    get_env_name(config,_id))
        os.makedirs(self.traces_folder)
        trace_filename = os.path.join(self.traces_folder, self.exp_name)
        self.viz = visdom.Visdom(env=self.exp_name,
                                 log_to_filename=trace_filename,
                                 **self.visdom_opts)
        self._log_env_url()

        config_str = json.dumps(config, sort_keys=True, indent=2, separators=(',', ': '))
        start_time_str = 'Started at {}'.format(utc_to_local(start_time))
        self.viz.text(f'{start_time_str}<pre>{config_str}</pre>', opts={
            'title': 'Config',
            'width': 350,
            'height': 700
        })
        logger.info(config_str)

    def completed_event(self, stop_time, result):
        local_time = utc_to_local(stop_time)
        logger.info('completed_event')
        delta = stop_time - self.start_time
        seconds = round(delta.total_seconds())
        run_time = datetime.timedelta(seconds=seconds)
        self.viz.text('Completed at {} after {}'.format(local_time, run_time))
        self._log_env_url()

    def interrupted_event(self, interrupt_time, status):
        local_time = utc_to_local(interrupt_time)
        logger.info('interrupted_event')

        self.viz.text('Interruped at {}'.format(local_time))
        self._log_env_url()

    def failed_event(self, fail_time, fail_trace):
        local_time = utc_to_local(fail_time)
        logger.info('failed_event')

        self.viz.text('Failed at {}\n{}'.format(local_time, fail_trace))
        self._log_env_url()

    def artifact_event(self, name, filename, metadata=None, content_type=None):
       logger.info('New artifact {}'.format(filename))

       if self._is_video(metadata):
           logger.info('Adding video "{}"'.format(name))
           opts = {
               'title': name,
           }
           if 'frame_size' in metadata:
               opts['width'] = metadata['frame_size'][1]
               opts['height'] = metadata['frame_size'][0]
           else:
               opts['width'] = 500
               opts['height'] = 500

           self.viz.video(videofile=filename, opts=opts)

    def heartbeat_event(self, info, captured_out, beat_time, result):
        # Ugly trick to set the pane name if using GNU Screen
        system("echo '\ek{}\e\\'".format(self.run_id))

    def _log_env_url(self):
        logger.info("http://{server}:{port}/env/{env}".format(**self.visdom_opts, env=self.exp_name))

    @staticmethod
    def _is_video(metadata):
        if metadata is None:
            return False
        if 'content-type' in metadata:
            return metadata['content-type'].startswith('video/')
        if 'content_type' in metadata:
            return metadata['content_type'].startswith('video/')
        return False
