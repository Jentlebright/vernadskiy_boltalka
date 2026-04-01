import os

from vernadskiy_boltalka.paths import project_root


def test_project_root_exists():
    root = project_root()
    assert os.path.isdir(root)


def test_project_root_contains_data_or_package():
    root = project_root()
    # в репозитории ожидаем каталог пакета и обычно data/ или vernadskiy_data/
    assert os.path.isdir(os.path.join(root, "vernadskiy_boltalka"))
