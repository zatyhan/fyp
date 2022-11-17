from rm_consts import remove_arrows
from gen_stack_dict import stacks

for idx, name in enumerate(stacks):
    print(name)
    paras_path = stacks[name]["paras_dir"]
    start_slice = stacks[name]["start_slice"]
    end_slice = stacks[name]["end_slice"]
    start_t = stacks[name]["start_t"]
    end_t = stacks[name]["end_t"]
    remove_arrows(start_slice, end_slice, start_t, end_t, paras_path)