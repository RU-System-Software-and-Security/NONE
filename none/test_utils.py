from torch.autograd import Variable
import torch

def test(model,testloader,num_classes,criterion,save_path,logger_file):
    model.eval()
    num = 0
    test_loss = 0
    correct = 0
    correct_class = [0]*num_classes
    index = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for i in range(data.shape[0]):
            for j in range(num_classes):
                if target[i].item() == j and pred[i] == j:
                    correct_class[j] = correct_class[j] + 1
                    
    test_loss /= len(testloader.dataset)
    torch.save(model.state_dict(), save_path + 'latest.pth')
    print('Clean Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    logger_file.write('Clean Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    for j in range(num_classes):
        print("class "+str(j)+" correct num: " + str(correct_class[j]))
        logger_file.write("class "+str(j)+" correct num: " + str(correct_class[j])+"\n")

def test_poisoned(model,testloader_poisoned,num_classes,criterion,save_path,logger_file,poison_type="single_target",attack_target=0):

    model.eval()
    num = 0
    test_loss = 0
    correct = 0
    num_correct_label_target_itself = 0
    num_label_target_itself = 0
    for data, target in testloader_poisoned:
        target_class_index_list = []
        for i in range(data.shape[0]):
            if target[i].item() == attack_target:
                target_class_index_list.append(i)
                num_label_target_itself = num_label_target_itself + 1
            if poison_type == "label_specific":
                if target[i] == (num_classes - 1):
                    target[i] = 0
                else:
                    target[i] = target[i] + 1
            else:
                target[i] = attack_target

        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        for index in target_class_index_list:
            if pred[index] == attack_target:
                num_correct_label_target_itself = num_correct_label_target_itself + 1
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    torch.save(model.state_dict(), save_path + 'latest.pth')
    correct_exclude_AtackTarget = correct - num_correct_label_target_itself
    print('Triggered Test set(exclude attack target itself): Accuracy: {}/{} ({:.2f}%)'.format(correct_exclude_AtackTarget, len(testloader_poisoned.dataset)-num_label_target_itself,100. * correct_exclude_AtackTarget / (len(testloader_poisoned.dataset)-num_label_target_itself)))
    print('Triggered Test set(include): Accuracy: {}/{} ({:.2f}%)'.format(correct, len(testloader_poisoned.dataset),100. * correct / len(testloader_poisoned.dataset)))
    logger_file.write('Triggered Test set(exclude attack target itself): Accuracy: {}/{} ({:.2f}%)\n'.format(correct_exclude_AtackTarget, len(testloader_poisoned.dataset)-num_label_target_itself,100. * correct_exclude_AtackTarget / (len(testloader_poisoned.dataset)-num_label_target_itself)))
    logger_file.write('Triggered Test set(include): Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(testloader_poisoned.dataset),100. * correct / len(testloader_poisoned.dataset)))

