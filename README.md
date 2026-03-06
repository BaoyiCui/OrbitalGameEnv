# OrbitalGameEnv

## 安装依赖
### 1.1 安装 vcpkg
```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```
将以下内容添加到`~/.bashrc`或者`~/.zshrc`
```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.2 配置 python 环境
```bash
conda create -n marppo python=3.13
conda activate marppo
pip install -r requirements.txt
```

### 1.3 在 CLion 中进行设置
找到 `settings -> Build, Execution, Deployment -> CMake`，
在右侧选中一个配置文件（一般是 `Debug` 或者 `Release`），在 `CMake options` 中添加如下内容
```bash
--toolchain <path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake -DPython_EXECUTABLE=<path-to-python>
```
