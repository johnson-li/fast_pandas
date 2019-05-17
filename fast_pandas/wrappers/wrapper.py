class Wrapper(object):
    __wrapped_class__ = None

    def __init__(self, *args, **kwargs):
        if not self.__wrapped_class__:
            raise NotImplemented('__wrapped_class__ is not specified')
        __wrapped_instance__ = kwargs.pop('__wrapped_instance__', None)
        if __wrapped_instance__:
            if not isinstance(__wrapped_instance__, self.__wrapped_class__):
                raise Exception('wrapped instance %s is not an instance of wrapped class %s' % (
                    __wrapped_instance__, self.__wrapped_class__))
            self.__wrapped_instance__ = __wrapped_instance__
            self.__wrapped_class__ = __wrapped_instance__.__class__
        else:
            self.__wrapped_instance__ = self.__wrapped_class__(*args, **kwargs)

    def __getattr__(self, item):
        orig_attr = self.__wrapped_instance__.__getattribute__(item)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                if result == self.__wrapped_instance__:
                    return self
                return result

            return hooked
        else:
            return orig_attr


class FunctionWrapper(object):
    def __init__(self, func):
        self.func = func

    def __getattr__(self, item):
        return self.func.__getattribute__(item)

    def __call__(self, *args, **kwargs):
        raise NotImplemented('a function wrapper should hook the original function by overwriting __call__')
