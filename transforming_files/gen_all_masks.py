from fill_chamber import fill_chamber
from gen_stack_dict import stacks

for idx, name in enumerate(stacks):
    start_slice = stacks[name]["start_slice"]
    end_slice = stacks[name]["end_slice"]
    start_t = stacks[name]["start_t"] - 1
    end_t = stacks[name]["end_t"] - 1
    case_path = stacks[name]["orig_dir"]
    paras_path = stacks[name]["paras_dir"]
    all_masked_path = paras_path + 'mask_with_chamber_new/'
    img_with_mask_path = paras_path + 'img_with_mask_new/'
    fill_chamber(start_slice, end_slice, start_t, 
                end_t, case_path, paras_path, 
                all_masked_path, img_with_mask_path)
    
print(f"There are {idx+1} different stacks")