from matplotlib import pyplot as plt

with open('logs/loss_list_full.txt','r') as log:
    dv_nr = 0
    batch_list = {dv_nr:[]}
    dv_loss_list = []
    dv_list = []
    epoch_list = []
    for logline in log:
        split_lst = logline.split(':')
        loss_value = float(split_lst[-1].split('\n')[0])
        #all batch losses:
        if split_lst[0] == 'Batch loss':
            batch_list[dv_nr].append(loss_value)
        #all mean-losses over dv
        else:
            dv_loss_list.append(loss_value)
            dv_nr += 1
            batch_list[dv_nr] = []

            dv_path = split_lst[0].split('/')[-1]
            epoch = int(dv_path.split('_')[1])
            dv = int(dv_path.split('_')[-1].split(')')[0])

            epoch_list.append(epoch)
            dv_list.append(dv)

# calculating the x-axis values to get integer as epochs and equal spaced values in between
dv_max = max(dv_list)+1
plot_epoch_list = []
for dv,epoch in zip(dv_list,epoch_list):  
    plot_epoch_list.append(epoch+(dv/dv_max))


plt.plot(plot_epoch_list,dv_loss_list,'.')
plt.xlabel('epochs in integer')
plt.ylabel('IoU-Loss')
plt.show()

