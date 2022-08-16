from ..data import transforms
from config4ml.data.transforms import Registry as T_registry

# Register all transforms
[T_registry.register(T) for T in transforms.get_all_transforms()]
print("Successfully registered transforms to config4ml!")
