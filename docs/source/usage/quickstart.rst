Quickstart
==========

First, import the package and simply import a task, stream, or stream generator ::

    import ctrl
    stream = ctrl.get_stream(n_tasks=10, size='+--+-')

this will create a new stream generator object, allowing to fetch tasks later.
This method contains all the boilerplate necessary for downloading, preprocessing, and saving the data to be able to reuse it later during the training process using your favorite Deep Learning framework.

