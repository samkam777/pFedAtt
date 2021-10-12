import os

def server_logging(epoch, total_train_loss, total_test_HR, total_test_NDCG, running_time, hyper_param):

    log_dir_total_train_loss = './' + running_time + "/" + hyper_param + "/" + '/logs/server/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    log_dir_total_test_HR = './' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_HR/'
    if not os.path.exists(log_dir_total_test_HR):
        os.makedirs(log_dir_total_test_HR)

    with open(log_dir_total_test_HR + 'total_test_HR.txt', 'a+') as f:
        f.write("%s\n" % total_test_HR)

    log_dir_total_test_NDCG = './' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_NDCG/'
    if not os.path.exists(log_dir_total_test_NDCG):
        os.makedirs(log_dir_total_test_NDCG)

    with open(log_dir_total_test_NDCG + 'total_test_NDCG.txt', 'a+') as f:
        f.write("%s\n" % total_test_NDCG)

    log_dir_total = './' + running_time + "/" + hyper_param + "/" + '/logs/server/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'global_total.txt', 'a+') as f:
        f.write("epoch %d " % epoch + "total_train_loss: %.4f " % total_train_loss + "total_test_HR: %.4f " % total_test_HR + "total_test_NDCG: %.4f \n" % total_test_NDCG)

def eps_logging(epoch, epsilons, running_time, hyper_param):

    log_dir_total_eps = './' + running_time + "/" + hyper_param + "/" + '/logs/total_epsilons/'
    if not os.path.exists(log_dir_total_eps):
        os.makedirs(log_dir_total_eps)

    with open(log_dir_total_eps + 'total_epsilons.txt', 'a+') as f:
        f.write("%s\n" % epsilons)

def user_logging(epoch, client_id, total_train_loss, total_test_HR, total_test_NDCG, running_time, hyper_param):

    log_dir_total_train_loss = './' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    log_dir_total_test_HR = './' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total_test_HR/'
    if not os.path.exists(log_dir_total_test_HR):
        os.makedirs(log_dir_total_test_HR)

    with open(log_dir_total_test_HR + 'total_test_HR.txt', 'a+') as f:
        f.write("%s\n" % total_test_HR)

    log_dir_total_test_NDCG = './' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total_test_NDCG/'
    if not os.path.exists(log_dir_total_test_NDCG):
        os.makedirs(log_dir_total_test_NDCG)

    with open(log_dir_total_test_NDCG + 'total_test_NDCG.txt', 'a+') as f:
        f.write("%s\n" % total_test_NDCG)

    log_dir_total = './' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'global_total.txt', 'a+') as f:
        f.write("epoch %d " % epoch + "user %d " % client_id + "total_train_loss: %.4f " % total_train_loss + "total_test_HR: %.4f " % total_test_HR + "total_test_NDCG: %.4f \n" % total_test_NDCG)










