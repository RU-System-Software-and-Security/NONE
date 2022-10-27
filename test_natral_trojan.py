import abs_test

def test_natral_trojan(save_path,image_size,arch,logger_file):
    for i in [4,5,6,7,8,9]:
        troj_size = (0.2-0.02*i)*image_size*image_size
        test_results = abs_test.run_abs(save_path+"latest.pth", "./abs/output", "./abs/output_naturebackdoor_cifar0411/scratch_all/scratch_epoch320/", "./abs/example", "./abs/output_naturebackdoor/log/", "log_epoch320.txt",  arch_local=arch, example_img_format='png', troj_size_override = troj_size)
        print(test_results)
        
        test_results_converted = []
        for item in test_results:
            item = item.split('_')[-1]
            test_results_converted.append(item)
        
        #show REASR under limited trigger size
        logger_file.write(str(troj_size)+":"+str(max(test_results_converted))+"\n")
        logger_file.flush()
        print(str(troj_size)+":"+str(max(test_results_converted))+"\n")