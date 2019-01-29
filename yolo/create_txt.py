import subprocess
import sys


def main():
    det_name = sys.argv[1]
    script_det = "./cfg/"+det_name+"/"+det_name
    video_path = "../video/"+sys.argv[2]+".mp4"

    cmd = ""
    if script_det is not None:
        cmd += script_det
    if video_path is not None:
        cmd += ' ' + video_path
    print(cmd)
    network = subprocess.Popen(cmd, shell = True)
    network.wait()


main()