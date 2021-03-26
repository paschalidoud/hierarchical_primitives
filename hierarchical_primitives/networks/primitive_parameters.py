import torch


class PrimitiveParameters(object):
    """Represents the \lambda_m."""
    def __init__(self, probs, translations, rotations, sizes, shapes,
                 space_partition=None, fit=None, qos=None, sharpness=None):
        self.probs = probs
        self.translations = translations
        self.rotations = rotations
        self.sizes = sizes
        self.shapes = shapes
        self.fit = fit
        self.space_partition = space_partition
        self.qos = qos
        self.sharpness = sharpness

    def __getattr__(self, name):
        if not name.endswith("_r"):
            raise AttributeError()

        prop = getattr(self, name[:-2])
        if not torch.is_tensor(prop):
            raise AttributeError()

        return prop.view(self.batch_size, self.n_primitives, -1)

    @property
    def members(self):
        return (
            self.probs,
            self.translations,
            self.rotations,
            self.sizes,
            self.shapes,
            self.space_partition,
            self.fit,
            self.qos,
            self.sharpness
        )

    @property
    def batch_size(self):
        return self.sizes.shape[0]

    @property
    def n_primitives(self):
        return self.sizes.shape[1] // 3

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]

    @classmethod
    def empty(cls):
        return cls(
            probs=None,
            translations=None,
            rotations=None,
            sizes=None,
            shapes=None,
            space_partition=None,
            fit=None,
            qos=None,
            sharpness=None
        )

    @classmethod
    def from_existing(cls, other, **kwargs):
        params = dict()
        params["probs"] = other.probs
        params["translations"] = other.translations
        params["rotations"] = other.rotations
        params["sizes"] = other.sizes
        params["shapes"] = other.shapes
        params["space_partition"] = other.space_partition
        params["fit"] = other.fit
        params["qos"] = other.qos
        params["sharpness"] = other.sharpness
        for key, value in list(kwargs.items()):
            if key in params:
                params[key] = value
        return cls(**params)

    @classmethod
    def with_keys(cls, **kwargs):
        p = cls.empty()
        return cls.from_existing(p, **kwargs)
