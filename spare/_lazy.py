"""Helpers for package exports backed by optional dependencies."""

from importlib import import_module


def lazy_exports(module_name, namespace, exports):
    def resolve(name):
        try:
            target_module, attribute = exports[name]
        except KeyError as exc:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from exc
        module = import_module(target_module)
        value = module if attribute is None else getattr(module, attribute)
        namespace[name] = value
        return value

    return resolve
