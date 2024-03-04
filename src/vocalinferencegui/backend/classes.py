from abc import ABC, abstractmethod
from pathlib import Path


class AttributeDict(dict):
    def __init__(self, data={}):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        data = self._reject_reserved_keys(data)
        super(AttributeDict, self).__init__(data)
        self.update(data)

    def _check_save_availability(self, cwd: dict):
        for i in cwd:
            if isinstance(cwd[i], dict):
                return self._check_save_availability(cwd[i])
            elif not isinstance(cwd[i], (str, int, float, bool, list)):
                return False
        return True

    def _reject_reserved_keys(self, object={}) -> dict:
        """reject all key prefixed with reserved string "__"

        Args:
                object (dict, optional): The dictionary you want to process. Defaults to {}.

        Returns:
                dict: the processed dictionary
        """
        reserved_key_prefix = "__"

        if isinstance(object, dict):
            for key, value in list(object.items()):
                if key.startswith(reserved_key_prefix):
                    del object[key]
                else:
                    object[key] = self._reject_reserved_keys(object[key])
        return object

    def make_json_able(self):
        """
        make the dict able to be converted to json
        """
        for i in self:
            if isinstance(self[i], dict):
                AttributeDict.make_json_able(self[i])
            elif isinstance(self[i], Path):
                self[i] = str(self[i])
        self.update(self)

    @classmethod
    def from_attrib_dict(cls, data):
        return cls(data.dict)

    @classmethod
    def from_path(cls, path: Path):
        from json import load

        return cls(load(open(path, "r")))

    @property
    def dict(self):
        return self.__dict__

    @staticmethod
    def tree(cwd: dict, layer=[]):
        for i in cwd:
            if isinstance(cwd[i], dict):
                AttributeDict.tree(cwd[i], layer + [i])
            else:
                print(".".join(layer + [i]), ":", cwd[i])

    @staticmethod
    def parse_point_expression(expression: str):
        """
        parse point expression to a list
        :param expression: point expression like "attr.subattr.subsubattr"
        :return: prased expression like ["attr", "subattr", "subsubattr"]
        """
        return expression.strip(".").split(".") if expression else []

    @staticmethod
    def construct_nested_dict(
        point_expression: list, value: object, point_dict={}
    ) -> dict:
        """
        construct a nested dict from a point expression.
        :param point_expression: point expression like "attr.subattr.subsubattr"
        :param value: value of the final value
        :param point_dict: initial value of the nested dict, must be {}
        :return:
        """
        if len(point_expression) == 1:
            point_dict[point_expression[0]] = value
        else:
            point_dict[point_expression[0]] = AttributeDict.construct_nested_dict(
                point_expression[1:], value, {}
            )
        return point_dict

    @staticmethod
    # add a value to a nested dict, according to the point expression, don't ignore different and nonexistent elements
    def add_to_dict(dict_1, point_expression: list, value: object, strict=True) -> dict:
        """
        add a value to a nested dict
        :param dict_1: base dict
        :param point_expression: point expression like "attr.subattr.subsubattr"
        :param value: value of the final value
        :param strict: whether check the attribute exist or not
        """
        AttributeDict.update_nested_dict(
            dict_1,
            AttributeDict.construct_nested_dict(point_expression, value, {}),
            strict,
        )
    def get_attribute(self, attr, cwd, strict=True) -> object:
        """
        get the reference of the value indicated in the position attr if the attribute exists and is a reference type
        :param attr: attribute pending fetch
        :param strict: whether check the attribute exist or not
        :param cwd: current working dict, or where to start with
        :return: a copy of the value indicated in the position attr
        """
        depth = 0
        if isinstance(attr, str):
            attr = AttributeDict.parse_point_expression(attr)
        if attr[0] not in cwd.keys():
            if strict:
                path = ""
                for i in range(depth + 1):
                    path += attr[i] + "."
                raise KeyError(f"Attribute {path.strip('.')} not found")
            else:
                return None
        else:
            if len(attr) == 1:
                return cwd.get(attr[0])
            else:
                return self.get_attribute(attr[1:], cwd=cwd.get(attr[0]), strict=strict)

    def set_attribute(self, attr: str, value: object, strict=True):
        """
        set the element at attr to value
        :param attr: attribute pending set
        :param value: value you wish to set
        :param strict: whether check the attribute exist or not
        :return:
        """
        attrs = AttributeDict.parse_point_expression(attr)
        self.update(
            AttributeDict.construct_nested_dict(attrs, value, {}), strict
        )
        return self

    def add_attribute(self, attr: str, value: object):
        return self.set_attribute(attr, value)

    def remove_attribute(self, attr: str, cwd, strict=True):
        """
        remove the element at attr
        :param attr: point expression of the element pending remove
        :param cwd: current working dictionary, the beginning of the search, usually self.__dict__
        :param strict: use strict mode or not, if not, raise exception when the attr doesn't exist
        """
        attrs = AttributeDict.parse_point_expression(attr)
        if AttributeDict(cwd)[attr] is None:
            if strict:
                raise KeyError(f"Attribute {attr} not found")
            return None
        elif len(attrs) == 1:
            del cwd[attrs[0]]
        else:
            attr = ""
            for i in attrs[1:]:
                attr += i + "."
            attr = attr.strip(".")
            self.remove_attribute(attr, cwd[attrs[0]], strict)

    def has_attribute(self, attr: str):
        return self.get_attribute(attr, self) is not None

    def to_file(self, name=None):
        if not self._check_save_availability(self.__dict__):
            raise ValueError("Cannot save this object")
        import pickle
        from json import dump

        if name is None or self.__dict__.get("name") is None:
            name = str(hash(pickle.dumps(self)) ** 2)[10:]
        else:
            name = self.__dict__.get("name")
        dump(self.__dict__, open(f"{name}.json", "w"))

    def update(self, entries, *args, **kwargs):
        for key, value in entries.items():
            if (isinstance(value, dict)):
                self.__dict__[key] = AttributeDict(value)
            else:
                self.__dict__[key] = value
                if (isinstance(value, list)):
                    self.__dict__[key] = self._update_list(value)
        self.__dict__.update(entries, *args, **kwargs)
    def _update_list(self, entries, *args, **kwargs):
        for i in range(len(entries)):
            if isinstance(entries[i], list):
                entries[i] = self._update_list(entries[i], *args, **kwargs)
            elif isinstance(entries[i], dict):
                entries[i] = AttributeDict(entries[i])
        return entries
    
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        self.set_attribute(key, value, strict=False)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __str__(self):
        """
        String value of the dictionary instance.
        """
        return str(self.__dict__)

    def __repr__(self):
        """
        String representation of the dictionary instance.
        """
        return repr(self.__dict__)

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__.keys())

    def __iter__(self):
        """
        Iterate over dictionary key/values.
        """
        return iter(self.__dict__.keys())

    def __len__(self):
        """
        Get number of items.
        """
        return len(self.__dict__.keys())

    def __contains__(self, key):
        """
        Check if key exists.
        """
        return self.__dict__.__contains__(key)

    def __reduce__(self):
        """
        Return state information for pickling.
        """
        return self.__dict__.__reduce__()

    def __eq__(self, other):
        """
        Check dictionary is equal to another provided dictionary.
        """
        return self.__dict__.__eq__(other)

    def __ne__(self, other):
        """
        Check dictionary is inequal to another provided dictionary.
        """
        return self.__dict__.__ne__(other)

    def __copy__(self):
        return AttributeDict(self.__dict__)


class ParamAbstract(ABC):
    from pathlib import Path

    def __init__(self, name: str, param_type: str, **kwargs):
        assert param_type in [
            "DemucsGenerate",
            "DemucsTrain",
            "SoVitsGenerate",
            "SoVitsTrain",
        ]
        self.data = AttributeDict()
        self.data.add_attribute("param_type", param_type)
        self.data.add_attribute("name", name)
        for key, value in kwargs.items():
            self.data.add_attribute(key, value)

    __dict__ = property(lambda self: self.data)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict):
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path: Path):
        pass

    @classmethod
    @abstractmethod
    def from_attribute_dict(cls, attr_dict: AttributeDict):
        return cls(attr_dict.name, attr_dict.type, **attr_dict.dict)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def save_as(self, name: str):
        pass

    @property
    def get(self):
        return self


class DemucsGenerateParam(ParamAbstract):
    def __init__(self, name: str, param_type: str, **kwargs):
        super().__init__(name, param_type, **kwargs)

    @property
    def name(self):
        return self.data["name"]

    def save_as(self, name: str = None):
        from environment import demucs_preset_path
        from json import dump

        dump(
            self.data.dict,
            open(
                f"{demucs_preset_path}/{self.data['name'] if name is None else name}.json",
                "w+",
            ),
        )

    @classmethod
    def from_dict(cls, data: dict):
        assert (
            data.get("param_type") == "DemucsGenerate" and data.get("name") is not None
        )
        return cls.from_attribute_dict(AttributeDict(data))

    @classmethod
    def from_list(cls, data: list, extension: str = "wav"):
        if data[-1] is None or data[-1] == "":
            data[-1] = str(hash(str(data)) ** 2)[10:]
        return cls.from_dict(
            {
                "param_type": "DemucsGenerate",
                "track_path": data[0],
                "output_path": data[2],
                "repo": data[4],
                "device": data[6],
                "wav_store_method": data[7].strip("--"),
                "split_mode": data[8].strip("--"),
                "clip_mode": data[10],
                "split_num": data[-3],
                "save_to_config": data[-2],
                "name": data[-1],
                "extension": extension,
            }
        )

    @classmethod
    def from_file(cls, name: str):
        from json import load
        from environment import demucs_preset_path

        return cls.from_dict(load(open(f"{demucs_preset_path}/{name}", "r")))

    @classmethod
    def from_attribute_dict(cls, attr_dict: AttributeDict):
        return DemucsGenerateParam(**attr_dict.dict)


class SoVitsGenerationParam(ParamAbstract):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "SoVitsGenerationParam", **kwargs)

    @property
    def name(self):
        pass

    @property
    def __str__(self):
        return
