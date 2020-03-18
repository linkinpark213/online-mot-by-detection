from typing import List, Dict


class Registry(object):
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._contents: Dict[str, object] = {}

    def register(self):
        def decorator(obj: object) -> object:
            assert obj.__name__ not in self._contents.keys(), "Key '{}' already exists in '{}' registry".format(
                obj.__name__, self._name
            )
            self._contents[obj.__name__] = obj
            return obj

        return decorator

    def get(self, name: str) -> object:
        assert name in self._contents.keys(), "Key '{}' doesn't exist in '{}' registry".format(
            name, self._name
        )
        return self._contents[name]

    def keys(self) -> List[str]:
        return list(self._contents.keys())

    def __repr__(self):
        s = self.__class__.__name__ + ' ' + self._name + '( '
        for key, value in self._contents:
            s += '\t' + key + ' => ' + value
        s += ')'
        return s
