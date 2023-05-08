## setuptools教程

### 什么是setuptools？

setuptools（包含easy_install）在distutils的基础上增加了一些改进。

setuptools是一个构建后端（build backend），用户看到的接口由pip, build这些工具提供的。

如果要使用setuptools，必须显式创建一个pyproject.toml文件。

### 安装最新版本的pip

```
python -m pip install --upgrade pip
```

### 项目结构

项目结构一般分为flat布局和src布局两种。

```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
└── example_package/
    ├── __init__.py
    ├── example.py
    ├── sub_package
    │   ├── __init__.py
    │   └── example2.py
    └── tests/
```


```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── example_package/
│       ├── __init__.py
│       └── example.py
└── tests/
```

### 创建pyproject.toml

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "example_package"
version = "0.0.1"
authors = [
  { name="Example Author", email="author@example.com" },
]
description = "A small example package"
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
"Homepage" = "https://github.com/pypa/sampleproject"
"Bug Tracker" = "https://github.com/pypa/sampleproject/issues"
```

build-system指定打包所使用的库。

初此之外，还需要指定一些和包有关的元数据、内容、依赖。建议直接在当前pyproject.toml中编辑，或者定义一个`setup.cfg`或者`setup.py`




### 生成发布文件

```
python -m pip install --upgrade build
python -m build  # 在pyproject.toml所在目录
```

于是生成dist目录，并包含如下文件：

```
dist/
├── example_package-0.0.1-py3-none-any.whl
└── example_package-0.0.1.tar.gz
```

### 上传发布文件

```
python -m install --upgrade twine
python -m twine upload dist/*
```



