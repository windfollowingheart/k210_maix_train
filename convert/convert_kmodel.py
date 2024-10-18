import subprocess
import argparse


def convert_to_kmodel(tf_lite_path, kmodel_path, ncc_path, images_path):
    '''
        @ncc_path ncc 可执行程序路径
        @return (ok, msg) 是否出错 (bool, str)
    '''
    p =subprocess.Popen([ncc_path, "-i", "tflite", "-o", "k210model", "--dataset", images_path, tf_lite_path, kmodel_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, err = p.communicate( )
        res = p.returncode
    except Exception as e:
        print("[ERROR] ", e)
        return False, str(e)
    res = p.returncode
    if res == 0:
        return True, "ok"
    else:
        print("[ERROR] ", res, output, err)
    return False, f"output:\n{output}\nerror:\n{err}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_lite_path", type=str, help="path to image file", default="")
    parser.add_argument("--kmodel_path", type=str, help="path to image file", default="")
    parser.add_argument("--ncc_path", type=str, help="path to image file", default="")
    parser.add_argument("--images_path", type=str, help="path to image file", default="")

    args = parser.parse_args()
    tf_lite_path=args.tf_lite_path
    kmodel_path=args.kmodel_path
    ncc_path=args.ncc_path
    images_path=args.images_path

    # tf_lite_path = "/kaggle/working/k210_maix_train/out/m.tflite"
    # kmodel_path = "/kaggle/working/k210_maix_train/out/m.kmodel"
    # ncc_path = "/kaggle/working/k210_maix_train/tools/ncc/ncc_v0.1/ncc"
    # images_path = "/kaggle/working/k210_maix_train/out/sample_images"

    is_kmodel_ok = convert_to_kmodel(tf_lite_path, kmodel_path, ncc_path, images_path)
    if is_kmodel_ok:
        print("convert kmodel Successfully")
    else:
        print("convert kmodel Failed")

