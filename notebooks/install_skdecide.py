from IPython import get_ipython
import glob
import json
import sys
import platform
import os
import shutil
import importlib

def install_skdecide(using_nightly_version=True, force_reinstall=False):
    if importlib.util.find_spec('scikit-decide') is not None and not force_reinstall:
        return
    
    on_colab = "google.colab" in str(get_ipython())

    if using_nightly_version:
        # remove previous installation
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        if os.path.exists('release.zip'):
            os.remove('release.zip')
        # look for nightly build download url
        release_curl_res = !curl -L -k -s -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/airbus/scikit-decide/releases/tags/nightly
        release_dict = json.loads(release_curl_res.s)
        release_download_url = sorted(
            release_dict["assets"], key=lambda d: d["updated_at"]
        )[-1]["browser_download_url"]
        print(release_download_url)

        # download and unzip
        !wget --output-document=release.zip {release_download_url}
        !unzip -o release.zip

        # get proper wheel name according to python version and platform used
        wheel_pythonversion_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        translate_platform_name = {'Linux': ' manylinux', 'Darwin': 'macosx', 'Windows': 'win'}
        platform_name = translate_platform_name[platform.system()]
        machine_name = platform.machine()
        wheel_path = glob.glob(
            f"dist/scikit_decide*{wheel_pythonversion_tag}*{platform_name}*{machine_name}.whl"
        )[0]

        skdecide_pip_spec = f"{wheel_path}\[all\]"
    else:
        skdecide_pip_spec = "scikit-decide\[all\]"

    if on_colab:
        # uninstall google protobuf conflicting with ray and sb3
        ! pip uninstall -y protobuf

    # install scikit-decide with all extras
    !pip --default-timeout=1000 install --upgrade --force-reinstall {skdecide_pip_spec}

    if on_colab:
        # be sure to load the proper cffi (downgraded compared to the one initially on colab)
        import cffi

        importlib.reload(cffi)
