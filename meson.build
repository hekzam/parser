project('hekzam-parser', 'cpp',
  version: '0.1.0',
  license: 'Apache-2.0',
  meson_version: '>=1.2.3'
)

opencv_dep = dependency('opencv4')
deps = [
  opencv_dep,
]

src = [
  'src/main.cpp'
]

parser = executable(
  'parser',
  src,
  dependencies: deps,
  install: true
)
