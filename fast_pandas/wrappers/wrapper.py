class Wrapper(object):
    __wrapped_class__ = None

    def __init__(self, *args, **kwargs):
        if not self.__wrapped_class__:
            raise NotImplemented('__wrapped_class__ is not specified')
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
