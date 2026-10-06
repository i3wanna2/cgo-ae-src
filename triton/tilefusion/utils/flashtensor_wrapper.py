import subprocess
import re
def get_flashtensor_ext_path(self, model: str, seqlen: int):
    # 1. execute the external script and capture output
    python_executable = "/root/miniconda3/envs/flashtensor/bin/python"
    script_path = "/home/meiziyuan/FlashTensor-AE/run_kernel.py"
    args = ["--model", model, "--system", "our", "--seqlen", str(seqlen)]
    output_log_file = f"/home/meiziyuan/triton/tilefusion/ft_logs/flashtensor_logs_{model}_{seqlen}.log"
    import torch 
    DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")
    is_a100 = False
    # if 'A100' in DEVICE_NAME:
    #     is_a100 = True
    #2. cmd
    # command = [python_executable, script_path] + args
    command = ["conda", "run", "-n", "flashtensor", "--no-capture-output", "python", script_path] + args
    cmd_str = f"source ~/flashtensor/bin/activate && python {script_path} {' '.join(args)}"
    try:
        if is_a100:
            print(f"running: {cmd_str} ...")
        else:
            print(f"running: {' '.join(command)} ...")
        
        # 3. run the command and capture output (stdout and stderr)
        if is_a100:
            # Use shell=True and executable='/bin/bash' to support 'source'
            result = subprocess.run(cmd_str, capture_output=True, text=True, shell=True, executable="/bin/bash")
        else :
            # text=True automatically converts the result to a string, check=True raises an exception if an error occurs
            result = subprocess.run(command, capture_output=True, text=True)
        full_output = result.stdout

        with open(output_log_file, "w") as f:
            f.write(full_output)
        print(f"save to : {output_log_file}")

        # 5. matching：write code to /tmp/xxxx.py
        pattern = r"write code to\s+(\/tmp\/\S+\.py)"
        match = re.search(pattern, full_output)

        if match:
            extracted_file = match.group(1)
            print(f"success match: {extracted_file}")
        else:
            print("not match")

    except subprocess.CalledProcessError as e:
        print(f"run fail: {e.returncode}")
        print(f"error msg: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return extracted_file