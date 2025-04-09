#!/usr/bin/env python
# coding: utf-8

''' 
This Script displays the image predictions for a loaded model

This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from utils import *

# TESS Dataset
class TESSDataset(Dataset):
    def __init__(self, angle_filename, ccd_folder, model_dir, image_shape, num_processes=20, above_sunshade=True, validation_dataset=True):
        start_time = time.time()
        
        # get data
        self.angle_folder = "/pdo/users/jlupoiii/TESS/data/angles/"
        self.ccd_folder = ccd_folder
        self.image_shape = image_shape
        self.model_dir = model_dir
        
        # Create a pool of processes
        pool = multiprocessing.Pool(processes=num_processes)

        # data matrices
        # X = []
        # Y = []
        # ffi_nums = []
        self.data = []
        self.labels = []
        self.ffi_nums = []

        # get angles_dic
        self.angles_dic = pickle.load(open(self.angle_folder+angle_filename, "rb"))

        print('dic is ', len(self.angles_dic.keys()))

        # get list of ffi's that were used as the validation set for the model
        validation_ffinumbers_list = pickle.load(open(self.model_dir+"validation_dataset_ffinumbers.pkl", "rb"))
        training_ffinumbers_list = pickle.load(open(self.model_dir+"training_dataset_ffinumbers.pkl", "rb"))
        validation_ffinumbers_set = set(validation_ffinumbers_list)
        training_ffinumbers_set = set(training_ffinumbers_list)
        print('length of set is', len(validation_ffinumbers_set), len(training_ffinumbers_set))
            
        files = []
        count = 0
        for filename in os.listdir(self.ccd_folder):
            ffi_num = filename[18:18+8]
            if ffi_num in self.angles_dic.keys(): # makes sure that fi_num exists
                if (self.angles_dic[ffi_num]['below_sunshade'] != above_sunshade) and (validation_dataset and ffi_num in validation_ffinumbers_set):
                    # above sunshade, in validation set
                    files.append(filename)
                elif (self.angles_dic[ffi_num]['below_sunshade'] != above_sunshade) and (not validation_dataset and ffi_num in training_ffinumbers_set):
                    # above sunshade, in training set
                    files.append(filename)
                elif (self.angles_dic[ffi_num]['below_sunshade'] != above_sunshade) and (validation_dataset and ffi_num in validation_ffinumbers_set):
                    # below sunshade, in validation set
                    files.append(filename)
                elif  (self.angles_dic[ffi_num]['below_sunshade'] != above_sunshade) and (not validation_dataset and ffi_num in training_ffinumbers_set):
                    # below sunshade, in training set
                    files.append(filename)

        print('length of dataset is', len(files))

        pbar_files = tqdm(files)
        results = []
        # print(f"About to start loading images to a list at time {(time.time() - start_time):.2f}")
        results = pool.map(self.load_images_worker, pbar_files)
        # print(f"made list of all processed results at time {(time.time() - start_time):.2f}")

        # Process the results
        pbar_results = tqdm(results)
        for x, y, ffi_num in pbar_results:
            if x is not None:
                self.data.append(x)
                self.labels.append(y)
                self.ffi_nums.append(ffi_num)
        
        pool.close()
        pool.join()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Time to build dataset of {len(self.data)} points: {total_time:.2f} seconds")

    def load_images_worker(self, filename):
        if len(filename) < 40 or filename[27] != '3': 
            return None, None, None

        image_arr = pickle.load(open(self.ccd_folder + filename, "rb"))
        ffi_num = filename[18:18+8]
        try:
            angles = self.angles_dic[ffi_num]
        except KeyError:
            return None, None, None
            
        x = np.array([angles['1/ED'], angles['1/MD'], angles['1/ED^2'], angles['1/MD^2'], angles['Eel'], angles['Eaz'], angles['Mel'], angles['Maz'], angles['E3el'], angles['E3az'], angles['M3el'], angles['M3az']])
        x = Image.fromarray(x)
        y = image_arr.flatten()
        y = Image.fromarray(y)

        return x, y, ffi_num


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        angles_image = self.data[idx]
        ffi_image = self.labels[idx]
        ffi_num = self.ffi_nums[idx]
        orbit = self.angles_dic[ffi_num]["orbit"]

        transform = transforms.Compose([
            transforms.ToTensor(),
            lambda s: s.reshape(1, 12)
        ])
        target_transform = transforms.Compose([
            lambda s: np.array(s),
            # lambda s: s.reshape((1024,1024)),
            lambda s: s.reshape(self.image_shape),
            transforms.ToTensor()
        ])

        angles_image = transform(angles_image)
        ffi_image = target_transform(ffi_image)

        # X: 1x12 vector of angles and distances
        # Y: 16x16 image (or other image size)
        return {"x":angles_image, "y":ffi_image, "ffi_num": ffi_num, "orbit": orbit}


# run function for evaluating a (single) model
def display_TESS_single_model():
    # hardcoding these here - make sure these match the parameters of the model being loaded
    # model parameters
    n_T = 600
    device = "cuda:5"
    n_feat = 256
    
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im128x128_multipleGPUs_1/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im64x64_multipleGPUs_split_orbits/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_EMabove_im64x64_multipleGPUs_splitOrbits_earlyStop/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_nonoversaturated_im64x64_multipleGPUs_splitOrbits_earlyStop/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im64x64_multipleGPUs_earlyStop/"
    model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im64x64_multipleGPUs_earlyStop_TEST/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im32x32_multipleGPUs_split_orbits/"
    # model_dir = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im64x64_multipleGPUs_1/"
    
    model_state_dic_filename = "model_epoch0.pth"
    eval_folder = 'eval/'
    os.makedirs(os.path.join(model_dir, eval_folder), exist_ok=True) # makes eval folder
    
    # dataset parameters - eventually make it so we load validation/training set that is saved specifically to the model
    # angle_filename = 'angles_O11-54_data_dic.pkl'
    angle_filename = 'angles_O11-54_data_dic.pkl'
    # ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im32x32/"
    ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im64x64/"
    # ccd_folder = "/pdo/users/jlupoiii/TESS/data/processed_images_im128x128/"
    # image_shape = (32,32)
    image_shape = (64,64)
    # image_shape = (128,128)
    num_processes = 40
    N = 10 # number of samples to predict per datapoint

    # loading class that handles getting the 4096x4096 images based on the ffi
    ffi_to_4096originalimage = TESS_4096_original_images()
    ffi_to_4096processedimage = TESS_4096_processed_images()

    torch.cuda.empty_cache()

    # load daatset and dataloader for images in validation set that are above the sunshade
    # only using the validation dataset
    dataset = TESSDataset(angle_filename=angle_filename, ccd_folder=ccd_folder, model_dir=model_dir, image_shape=image_shape, num_processes=num_processes, above_sunshade=True, validation_dataset=True)
    # dataset = valid_above_dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5, drop_last=True)

    # define model
    in_dim = next(iter(dataloader))['x'].shape[2]
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, in_dim=in_dim,n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    
    # load a model
    ddpm.load_state_dict(torch.load(os.path.join(model_dir, model_state_dic_filename)))
    ddpm.to(device)
    print('model directory:', os.path.join(model_dir, model_state_dic_filename))

    # tracks losses and standard deviations for each image to see average loss
    losses = []
    sigmas = []
    # var_scaled_RMSEs = [] # used for errenuous calculations of var scaled rmse, will delete later
    var_scaled_RMSE_pixels_list = np.array([])


    '''
    ################ scatterplots ##################
    print('starting to make pairplot')
    dataset_training = TESSDataset(angle_filename=angle_filename, ccd_folder=ccd_folder, model_dir=model_dir, image_shape=image_shape, num_processes=num_processes, above_sunshade=True, validation_dataset=False)
    # making and saving pairplot of input data - separates training and validation data
    data = []
    for i in range(len(dataset_training)):
        data.append(dataset_training[i]['x'].flatten().numpy().astype(float))
    for i in range(len(dataset)):
        data.append(dataset[i]['x'].flatten().numpy().astype(float))
    df = pd.DataFrame(data)
    # df.columns = ['1/ED','1/MD','1/ED^2','1/MD^2','Eel','Eaz','Mel','Maz','E3el','E3az','M3el','M3az']
    df.columns = [r"$\frac{1}{d_E}$",r"$\frac{1}{d_M}$",r"$(\frac{1}{d_E})^2$",r"$(\frac{1}{d_M})^2$",r"$\theta_{E,sunshade}$",r"$\phi_{E,sunshade}$",r"$\theta_{M,sunshade}$",r"$\phi_{M,sunshade}$",r"$\theta_{E,camera}$",r"$\phi_{E,camera}$",r"$\theta_{M,camera}$",r"$\phi_{M,camera}$"]
    
    # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    sns.scatterplot(data=df[:len(dataset_training)], x=r"$(\frac{1}{d_M})^2$", y=r"$\theta_{M,camera}$", color='#1f77b4', alpha=0.4, s=10, ax=axs[0], label='Training')
    sns.scatterplot(data=df[len(dataset_training):], x=r"$(\frac{1}{d_M})^2$", y=r"$\theta_{M,camera}$", color='#ff7f0e', alpha=0.4, s=10, ax=axs[0], label='Validation')
    sns.scatterplot(data=df[:len(dataset_training)], x=r"$\phi_{E,camera}$", y=r"$\theta_{M,camera}$", color='#1f77b4', alpha=0.4, s=10, ax=axs[1])
    sns.scatterplot(data=df[len(dataset_training):], x=r"$\phi_{E,camera}$", y=r"$\theta_{M,camera}$", color='#ff7f0e', alpha=0.4, s=10, ax=axs[1])
    sns.scatterplot(data=df[:len(dataset_training)], x=r"$(\frac{1}{d_M})^2$", y=r"$\phi_{M,camera}$", color='#1f77b4', alpha=0.4, s=10, ax=axs[2])
    sns.scatterplot(data=df[len(dataset_training):], x=r"$(\frac{1}{d_M})^2$", y=r"$\phi_{M,camera}$", color='#ff7f0e', alpha=0.4, s=10, ax=axs[2])
    sns.scatterplot(data=df[:len(dataset_training)], x=r"$\frac{1}{d_M}$", y=r"$\phi_{M,sunshade}$", color='#1f77b4', alpha=0.4, s=10, ax=axs[3])
    sns.scatterplot(data=df[len(dataset_training):], x=r"$\frac{1}{d_M}$", y=r"$\phi_{M,sunshade}$", color='#ff7f0e', alpha=0.4, s=10, ax=axs[3])
    axs[0].legend().remove()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=14)
    for ax in axs:
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)    
    plt.tight_layout()
    plt.show()    
    fig.savefig(os.path.join(model_dir, "data_pairplot_partial.png"), bbox_inches='tight')
    plt.close()
    return
    
    source = ['Training'] * len(dataset_training) + ['Validation'] * len(dataset)
    df['source'] = source
    sns.set(font_scale=1.2, style="ticks")
    colors = ['#1f77b4', '#ff7f0e']
    sns.set_palette(sns.color_palette(colors))
    pairplot = sns.pairplot(df, hue='source', height=1.5, corner=True,plot_kws={"s": 3, "alpha":0.5}, diag_kind=None)
    pairplot._legend.set_title(' ')
    pairplot._legend.set_bbox_to_anchor((.6, .6))
    pairplot_fig = pairplot.fig
    # plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
    pairplot_fig.savefig(os.path.join(model_dir, "data_pairplot_full.png"))
    plt.show()
    plt.close()
    return

    KL_train = []
    KL_valid = []
    KL_divergences = np.zeros((12, 12))
    for i in range(len(dataset_training)):
        KL_train.append(dataset_training[i]['x'].flatten().numpy().astype(float))
    for i in range(len(dataset)):
        KL_valid.append(dataset[i]['x'].flatten().numpy().astype(float))
    KL_train = np.array(KL_train).T
    KL_valid = np.array(KL_valid).T
    # print(KL_train.shape)
    for i in range(12):
        for j in range(12):
            print(KL_train[i].shape)
            KL_divergences[i, j] = entropy(KL_train[i], KL_valid[j])
            # KL_divergences[i, j] = entropy(KL_train[i], KL_train[i])
    print(KL_divergences)
    return
    ################ scatterplots ##################
    '''

    torch.cuda.empty_cache()
    ddpm.eval()
    with torch.no_grad():
        count = 0
        
        for data_dic in dataloader:

            torch.cuda.empty_cache()

        
            ffi_num = data_dic['ffi_num'][0]
            # if ffi_num != '00018464': continue

            orbit = data_dic['orbit'][0]
            x = data_dic['y'].to(device) # .to(device) has to do with CPU and the GPU
            c = data_dic['x'].to(device)

            # To display processed image
            # img = plt.imshow(x[0][0].cpu() * 633118, cmap='gray', vmin=0, vmax=633118)
            # plt.title(f"Processed Image\nffi {ffi_num}")
            # # plt.axis('off')
            # plt.colorbar(img, fraction=0.04)
            # plt.show()
            # plt.close()
            # continue

            torch.cuda.empty_cache()

            print(f'beginnging evaluation for ffi number {ffi_num}, orbit {orbit}')
    
            # generate samples using model, each sample is an x_i
            x_gen, x_gen_store = ddpm.sample_c(c, N, (1, image_shape[0], image_shape[1]), device)

            # mid_time = time.time()
            
            # get original 4096x4096 image
            X = ffi_to_4096originalimage[ffi_num]

            # get processed 4096x4096 image
            X_proc_4096 = ffi_to_4096processedimage[ffi_num]
            X_proc_4096 = torch.tensor(X_proc_4096)

            '''
            ############## makes GIFS #############
            # only animates a single sample(don't need 100!)
            x_gen_store = x_gen_store[:, 0:1, :, :, :]
            # create gif of images evolving over time, based on x_gen_store
            fig, axs = plt.subplots(nrows=1, ncols=1,sharex=True,sharey=True, figsize=(8,8))# ,figsize=(8,3))
            def animate_diff(i, x_gen_store):
                print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                img = axs.imshow(x_gen_store[i, 0, 0], cmap='gray', vmin=0, vmax=1)
                axs.set_xticks([])
                axs.set_yticks([])
                return img,
            ani = FuncAnimation(fig, animate_diff, fargs=[np.clip(x_gen_store, 0, 1)],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0]) 
            save_file = os.path.join(model_dir, eval_folder, f"gif_sample_{ffi_num}_{count}.gif")
            ani.save(save_file, dpi=100, writer=PillowWriter(fps=5))
            print('Done animating GIF')
            ############## makes GIFS (end) #############
            '''

            # taking mean and std dev of all predictions (downsampled)
            bar_x = torch.mean(x_gen.cpu(), dim=0) # Mean prediction for model's predictions
            sigma_x = torch.std(x_gen.cpu(), dim=0) # Standard deviation of prediction


            '''
            #### Variance scaled error calculations for DOWNSAMPLED data, makes histogram ####
            starting = time.time()
            var_scaled_RMSE_pixels = ((bar_x[0] - x.cpu()) / sigma_x[0]).flatten().numpy()
            var_scaled_RMSE_pixels_list = np.append(var_scaled_RMSE_pixels_list, var_scaled_RMSE_pixels)
            plt.hist(var_scaled_RMSE_pixels_list, bins=100, density=True, range=(-5, 5))
            plt.xlim(-5, 5)
            plt.xlabel('Uncertainty Scaled RMSE', fontsize=14)
            temp = np.linspace(-5, 5, 100)
            normal_curve = norm.pdf(temp, 0, 1)
            plt.plot(temp, normal_curve, 'k--', linewidth=2)  # Dashed black line
            plt.show()
            plt.savefig(os.path.join(model_dir, eval_folder, f'uncertainty_scaled_error_downsampled_N{N}_{model_state_dic_filename.split("_")[1]}.pdf'))
            # print('saved temp hist')
            plt.close()
            print(f'Time in seconds to make histogram: {time.time()-starting}')
            #### Variance scaled error calculations for DOWNSAMPLED data, makes histogram  (end) ######
            continue
            '''
            
            

            # upscale generated samples. Using linear interpolation
            order = 1 # degree of interpolation. 1 is linear
            x_gen_upsampled = scipy.ndimage.zoom(x_gen.cpu(), (1, 1, 4096//image_shape[0], 4096//image_shape[0]), order=order)
            x_gen_upsampled = torch.tensor(x_gen_upsampled)

            # taking mean and std dev of all upsampled predictions
            bar_x_upsampled = torch.mean(x_gen_upsampled, dim=0) # Mean prediction for upsampled predictions
            sigma_x_upsampled = torch.std(x_gen_upsampled, dim=0) # Standard deviation of upsampled prediction

            

            
            #### Variance scaled error calculations for UPSAMPLED data, makes histogram ####
            starting = time.time()
            var_scaled_RMSE_pixels = ((bar_x_upsampled[0] - X_proc_4096) / sigma_x_upsampled[0]).flatten().numpy() * N**0.5
            var_scaled_RMSE_pixels_list = np.append(var_scaled_RMSE_pixels_list, var_scaled_RMSE_pixels)
            plt.hist(var_scaled_RMSE_pixels_list, bins=100, density=True, range=(-5, 5), color='orange')
            plt.xlim(-5, 5)
            plt.xlabel('Uncertainty-scaled RMSE', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            temp = np.linspace(-5, 5, 100)
            normal_curve = norm.pdf(temp, 0, 1)
            plt.plot(temp, normal_curve, 'k--', linewidth=2, label="Standard norm")  # Dashed black line
            plt.legend()
            plt.show()
            # plt.savefig(os.path.join(model_dir, eval_folder, f'uncertainty_scaled_error_N{N}_{model_state_dic_filename.split("_")[1]}.pdf'))
            plt.savefig(os.path.join(model_dir, eval_folder, f'uncertainty_scaled_error_N{N}_{model_state_dic_filename.split("_")[1]}_new.png'))
            plt.close()
            print(f'Time in seconds to make histogram: {time.time()-starting}')
            continue
            #### Variance scaled error calculations for UPSAMPLED data, makes histogram  (end) ######
            
            
            
            losses.append(torch.mean(bar_x_upsampled))
            sigmas.append(torch.max(sigma_x_upsampled))
            print('average losses so far', np.mean(losses))
            print('average max std dev so far', np.mean(sigmas))

            '''
            ############# Displays evaluation images, separately ###########
            # Original Image
            # Processed Image
            # Prediction Image (just plot the mean)
            img = plt.imshow(X, cmap='gray', vmin=0, vmax=1)
            plt.title(f"Original Image\nffi {ffi_num}")
            # plt.axis('off')
            plt.colorbar(img, fraction=0.04)
            plt.show()
            plt.close()
            img = plt.imshow(x[0][0].cpu() * 633118, cmap='gray', vmin=0, vmax=633118)
            plt.title(f"Processed Image\nffi {ffi_num}")
            # plt.axis('off')
            plt.colorbar(img, fraction=0.04)
            plt.show()
            plt.close()
            img = plt.imshow(x_gen[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Model Prediction\nffi {ffi_num}")
            # plt.axis('off')
            plt.colorbar(img, fraction=0.04)
            plt.show()
            plt.close()
            img = plt.imshow(x_gen_upsampled[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
            # plt.title(f"Upsampled Prediction\nffi {ffi_num}")
            # plt.axis('off')
            plt.colorbar(img, fraction=0.04)
            plt.show()
            plt.close()
            img = plt.imshow(x_gen_upsampled[0][0].cpu() * 633118, cmap='gray', vmin=0, vmax=633118)
            plt.title(f"Upsampled Scaled Prediction\nffi {ffi_num}")
            # plt.axis('off')
            plt.colorbar(img, fraction=0.04)
            plt.show()
            plt.close()
            ############# Displays evaluation images, separately (end)###########
            '''

            '''
            ################## Print 4 images separately for poster ##################
            fig,ax = plt.subplots()
            im = ax.imshow(X, cmap="gray", vmin=0, vmax=1)
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_results_original.png", format="png")
            plt.close()

            fig,ax = plt.subplots()
            im = ax.imshow(bar_x_upsampled[0], cmap="gray", vmin=0, vmax=1)
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_results_mean.png", format="png")
            plt.close()

            fig,ax = plt.subplots()
            im = ax.imshow(sigma_x_upsampled[0], cmap="gray")
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_results_std.png", format="png")
            plt.close()

            fig,ax = plt.subplots()
            im = ax.imshow(X-bar_x_upsampled[0], cmap="gray", vmin=0, vmax=1)
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_results_corrected.png", format="png")
            plt.close()
            print('saved the images')
            return
            ################## Print 4 images separately for poster (end) ##################
            '''

            '''
            ################## Print 2 more images separately for poster ##################
            fig,ax = plt.subplots()
            im = ax.imshow(x_gen[0][0].cpu(), cmap="gray", vmin=0, vmax=1)
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_postprocessing_downsampled.png", format="png")
            plt.close()

            fig,ax = plt.subplots()
            im = ax.imshow(x_gen_upsampled[0][0].cpu(), cmap="gray", vmin=0, vmax=1)
            plt.grid(visible=False)
            colorbar = fig.colorbar(im)
            plt.savefig("/pdo/users/jlupoiii/TESS/poster_images/image_postprocessing_upsampled.png", format="png")
            plt.close()
            ################## Print 2 more images separately for poster ##################
            '''
            
            '''
            ####### plots 10 evaluation images in a neat grid, used in presentation ##########
            # make plot of each image 
            fig, axes = plt.subplots(2, 5, figsize=(18, 12)) # , subplot_kw={'aspect': 'equal'})
            fig.suptitle(f"{image_shape[0]}x{image_shape[1]} Model Performance\nOrbit {orbit}, ffi number {ffi_num}\n{N} sample predictions\nAverage Prediction MSE Loss: {avg_pred_RMSE_Loss:.2e}", fontsize = 25)

            # X
            img0 = axes[0,0].imshow(X, cmap='gray', vmin=0, vmax=1)
            axes[0,0].set_title(r"$X$" + "\nOriginal Image")
            fig.colorbar(img0, ax=axes[0,0], fraction=0.04)

            # X_proc_4096
            img1 = axes[0,1].imshow(X_proc_4096, cmap='gray', vmin=0, vmax=1)
            axes[0,1].set_title(r"$X_{proc,4096}$" + "\n4096x4096 Processed Image")
            fig.colorbar(img1, ax=axes[0,1], fraction=0.04)

            # X-X_proc_4096
            img2 = axes[0,2].imshow(X-X_proc_4096, cmap='gray', vmin=0, vmax=1)
            axes[0,2].set_title(r"$X-X_{proc,4096}$" + "\nProcessed Removed\nScattered Light")
            fig.colorbar(img2, ax=axes[0,2], fraction=0.04)

            # X_proc
            img3 = axes[0,3].imshow(x[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
            axes[0,3].set_title(r"$X_{proc}$" + f"\n{image_shape[0]}x{image_shape[0]} Processed Image")
            fig.colorbar(img3, ax=axes[0,3], fraction=0.04)
            
            # bar(x)
            img4 = axes[0,4].imshow(bar_x[0], cmap='gray', vmin=0, vmax=1)
            axes[0,4].set_title(r"$\bar{x}$" + "\nMean Prediction")
            fig.colorbar(img4, ax=axes[0,4], fraction=0.04)

            # sigma_x
            img5 = axes[1,0].imshow(sigma_x[0], cmap='gray')
            axes[1,0].set_title(r"$\sigma_x$" + "\nStandard Deviation\nof Prediction")
            fig.colorbar(img5, ax=axes[1,0], fraction=0.04)

            # bar(x)_up
            img6 = axes[1,1].imshow(bar_x_upsampled[0], cmap='gray', vmin=0, vmax=1)
            axes[1,1].set_title(r"$\bar{x}_{up}$" + "\nMean Upsampled Prediction")
            fig.colorbar(img6, ax=axes[1,1], fraction=0.04)
            
            # sigma_x_up
            img7 = axes[1,2].imshow(sigma_x_upsampled[0], cmap='gray')
            axes[1,2].set_title(r"$\sigma_{x, up}$" + "\nStandard Deviation\nof Upsampled Prediction")
            fig.colorbar(img7, ax=axes[1,2], fraction=0.04)

            # X_proc - bar(x)_up
            img8 = axes[1,3].imshow(X_proc_4096-bar_x_upsampled[0], cmap='gray')
            axes[1,3].set_title(r"$X_{proc,4096} - \bar{x}_{up}$" + "\nMean Upsampled Prediction Error")
            fig.colorbar(img8, ax=axes[1,3], fraction=0.04)

            # # X_proc - bar(x)_up
            # img8 = axes[1,3].imshow(X_proc-bar_x_upsampled[0], cmap='gray')
            # axes[1,3].set_title(r"$X_{proc} - \bar{x}_{up}$" + "\nMean Upsampled Prediction Error\n(colorbar scaled)")
            # fig.colorbar(img8, ax=axes[1,3], fraction=0.04)

            # X - bar(x)
            img9 = axes[1,4].imshow(X-bar_x_upsampled[0], cmap='gray', vmin=0, vmax=1)
            axes[1,4].set_title(r"$X - \bar{x}_{up}$" + "\nMean Upsampled Predicted\nRemoved Scattered Light")
            fig.colorbar(img9, ax=axes[1,4], fraction=0.04)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            # save images
            plt.tight_layout()
            fig.savefig(os.path.join(model_dir, eval_folder, f"eval_eval_{ffi_num}_{count}.pdf"))
            print('saved image at ' + os.path.join(model_dir, eval_folder, f"eval_{ffi_num}_{count}.pdf"))
            plt.close()
            ####### plots 10 evaluation images in a neat grid, used in presentation (end) ##########
            '''




            '''
            ####### plots 7 evaluation images in a neat grid, used in paper (end) ##########
            # make plot of each image 
            fig, axes = plt.subplots(2, 4, figsize=(18, 9)) # , subplot_kw={'aspect': 'equal'})
            # fig.suptitle(f"{image_shape[0]}x{image_shape[1]} Model Performance\nOrbit {orbit}, ffi number {ffi_num}\n{N} sample predictions\nAverage Prediction MSE Loss: {avg_pred_RMSE_Loss:.2e}", fontsize = 25)

            # X
            img0 = axes[0,0].imshow(X*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[0,0].set_title(r"$X_0$")
            fig.colorbar(img0, ax=axes[0,0], fraction=0.04)

            # X_proc_4096
            img1 = axes[0,1].imshow(X_proc_4096*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[0,1].set_title(r"$X_{proc,4096}$")
            fig.colorbar(img1, ax=axes[0,1], fraction=0.04)

            # X-X_proc_4096
            img2 = axes[0,2].imshow((X-X_proc_4096)*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[0,2].set_title(r"$X_0-X_{proc,4096}$")
            fig.colorbar(img2, ax=axes[0,2], fraction=0.04)

            # X_proc
            img3 = axes[0,3].imshow(x[0][0].cpu()*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[0,3].set_title(r"$X_{proc}$")
            fig.colorbar(img3, ax=axes[0,3], fraction=0.04)

            # bar(x)_up
            img6 = axes[1,0].imshow(bar_x_upsampled[0]*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[1,0].set_title(r"$\bar{x}_{up}$")
            fig.colorbar(img6, ax=axes[1,0], fraction=0.04)
            
            # sigma_x_up
            img7 = axes[1,1].imshow(sigma_x_upsampled[0], cmap='gray')
            axes[1,1].set_title(r"$\sigma_{x, up}$")
            fig.colorbar(img7, ax=axes[1,1], fraction=0.04)

            # X_proc - bar(x)_up
            img8 = axes[1,2].imshow((X_proc_4096-bar_x_upsampled[0])*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[1,2].set_title(r"$X_{proc,4096} - \bar{x}_{up}$")
            fig.colorbar(img8, ax=axes[1,2], fraction=0.04)

            # X - bar(x)
            img9 = axes[1,3].imshow((X-bar_x_upsampled[0])*633118, cmap='gray', vmin=0, vmax=1*633118)
            axes[1,3].set_title(r"$X_0 - \bar{x}_{up}$")
            fig.colorbar(img9, ax=axes[1,3], fraction=0.04)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            # save images
            plt.tight_layout()
            fig.savefig(os.path.join(model_dir, eval_folder, f"eval_{ffi_num}_{count}.pdf"))
            print('saved image at ' + os.path.join(model_dir, eval_folder, f"eval_{ffi_num}_{count}.pdf"))
            plt.close()
            ####### plots 10 evaluation images in a neat grid, used in presentation (end) ##########
            '''

            count += 1
            # if count == 100: break
            

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    display_TESS_single_model()
    
    


'''
########## code for comparing two different models of 2 different sizes ###############
# can ignore this
def display_TESS_two_models():

    # directory to which files are being saved
    eval_folder_path = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im128x128_multipleGPUs_1/eval_compare"
    os.makedirs(eval_folder_path, exist_ok=True)

    # hardcoding these here - make sure these match the parameters of the model being loaded
    # model parameters for 64x64 model
    n_T_64 = 600
    device_64 = "cuda:5"
    n_feat_64 = 256
    lrate_64 = 1e-4
    model_dir_64 = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im64x64_multipleGPUs_1/"
    model_state_dic_filename_64 = "model_epoch1499.pth"    
    angle_filename_64 = 'angles_O11-54_data_dic.pkl'
    ccd_folder_64 = "/pdo/users/jlupoiii/TESS/data/processed_images_im64x64/"
    image_shape_64 = (64,64)
    num_processes_64 = 5

    # hardcoding these here - make sure these match the parameters of the model being loaded
    # model parameters for 128x128 model
    n_T_128 = 600
    device_128 = "cuda:6"
    n_feat_128 = 256
    lrate_128 = 1e-4
    model_dir_128 = "/pdo/users/jlupoiii/TESS/model_conditional_diffusion/model_TESS/model_TESS_O11-54_im128x128_multipleGPUs_1/"
    model_state_dic_filename_128 = "model_epoch1499.pth"    
    # dataset parameters
    angle_filename_128 = 'angles_O11-54_data_dic.pkl'
    ccd_folder_128 = "/pdo/users/jlupoiii/TESS/data/processed_images_im128x128/"
    image_shape_128 = (128,128)
    num_processes_128 = 5

    N = 100 # number of samples to predict per datapoint

    # loading class that handles getting the 4096x4096 images based on the ffi
    ffi_to_4096originalimage = TESS_4096_original_images()

    # define models
    ddpm_64 = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat_64), betas=(1e-4, 0.02), n_T=n_T_64, device=device_64, drop_prob=0.1)
    ddpm_128 = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat_128), betas=(1e-4, 0.02), n_T=n_T_128, device=device_128, drop_prob=0.1)
    
    # load a model
    ddpm_64.load_state_dict(torch.load(os.path.join(model_dir_64, model_state_dic_filename_64)))
    ddpm_64.to(device_64)
    ddpm_128.load_state_dict(torch.load(os.path.join(model_dir_128, model_state_dic_filename_128)))
    ddpm_128.to(device_128)

    print(f"loading models from {os.path.join(model_dir_64, model_state_dic_filename_64)} and {os.path.join(model_dir_64, model_state_dic_filename_64)}")

    # load daatset and dataloader for images in validation set that are above the sunshade
    dataset_64 = TESSDataset(angle_filename=angle_filename_64, ccd_folder=ccd_folder_64, model_dir=model_dir_64, image_shape=image_shape_64, num_processes=num_processes_64, above_sunshade=True, validation_dataset=True)
    dataloader_64 = DataLoader(dataset_64, batch_size=1, shuffle=True, num_workers=5, drop_last=True)
    dataset_128 = TESSDataset(angle_filename=angle_filename_128, ccd_folder=ccd_folder_128, model_dir=model_dir_128, image_shape=image_shape_128, num_processes=num_processes_128, above_sunshade=True, validation_dataset=True)
    dataloader_128 = DataLoader(dataset_128, batch_size=1, shuffle=True, num_workers=5, drop_last=True)

    ddpm_64.eval()
    ddpm_128.eval()
    with torch.no_grad():
        count = 0        
        for data_dic_128 in dataloader_128:
            ffi_num = data_dic_128['ffi_num'][0]
            orbit = data_dic_128['orbit'][0]

            # ensures that the images are the same that are used in both the 64 and 128 model.
            # only a fraction of images are above the sunshade and in the validation set, so
            # the datasets may have many differing ffi_nums
            data_dic_64 = {}
            for data_dic_64_temp in dataloader_64:
                if data_dic_64_temp['ffi_num'][0] == ffi_num: # matching ffi_num was found!
                    data_dic_64 = data_dic_64_temp
                    break
            if not data_dic_64: # in the case that no matching ffi number was found
                continue
                
            x_128 = data_dic_128['y'].to(device_128)
            c_128 = data_dic_128['x'].to(device_128)
            x_64 = data_dic_64['y'].to(device_64)
            c_64 = data_dic_64['x'].to(device_64)

            print(f'beginnging evaluation for ffi number {ffi_num}, orbit {orbit}')
    
            # get original 4096x4096 image
            X = ffi_to_4096originalimage[ffi_num] # Original Image

            # for clarity, X_proc is the processed image to only show scattered light
            order = 1 # degree of interpolation. 1 is linear, what we want to do
            X_proc_64 = x_64[0].cpu()
            X_proc_upsampled_64 = scipy.ndimage.zoom(X_proc_64.cpu(), (1, 4096//image_shape_64[0], 4096//image_shape_64[0]), order=order)
            X_proc_upsampled_64 = torch.tensor(X_proc_upsampled_64)
            X_proc_128 = x_128[0].cpu()
            X_proc_upsampled_128 = scipy.ndimage.zoom(X_proc_128.cpu(), (1, 4096//image_shape_128[0], 4096//image_shape_128[0]), order=order)
            X_proc_upsampled_128 = torch.tensor(X_proc_upsampled_128)
    
            # generate samples using model, each sample is an x_i
            x_gen_64, x_gen_store_64 = ddpm_64.sample_c(c_64, N, (1, image_shape_64[0], image_shape_64[1]), device_64)
            x_gen_128, x_gen_store_128 = ddpm_128.sample_c(c_128, N, (1, image_shape_128[0], image_shape_128[1]), device_128)
            
            # upscale generated samples. Here, simply repeating pixels to represent the larger area
            x_gen_upsampled_64 = scipy.ndimage.zoom(x_gen_64.cpu(), (1, 1, 4096//image_shape_64[0], 4096//image_shape_64[0]), order=order)
            x_gen_upsampled_64 = torch.tensor(x_gen_upsampled_64)
            x_gen_upsampled_128 = scipy.ndimage.zoom(x_gen_128.cpu(), (1, 1, 4096//image_shape_128[0], 4096//image_shape_128[0]), order=order)
            x_gen_upsampled_128 = torch.tensor(x_gen_upsampled_128)
            
            bar_x_64 = torch.mean(x_gen_upsampled_64, dim=0) # Mean prediction for upsampled predictions
            bar_x_128 = torch.mean(x_gen_upsampled_128, dim=0)

            # bar_x_gen = torch.mean(x_gen, dim=0) # Mean prediction for non-upsampled predictions

            sigma_x_64 = torch.std(x_gen_upsampled_64, dim=0) # Standard deviation of prediction
            sigma_x_128 = torch.std(x_gen_upsampled_128, dim=0)

            avg_pred_RMSE_Loss_64 = torch.std((X_proc_upsampled_64 - bar_x_64)[0]) # MSE for average predicted image
            avg_pred_RMSE_Loss_128 = torch.std((X_proc_upsampled_128 - bar_x_128)[0])
            
            # make plot of each image 
            fig, axes = plt.subplots(2, 7, figsize=(18, 12)) # , subplot_kw={'aspect': 'equal'})

            fig.suptitle(f"Model Performance\nOrbit {orbit} , FFI Number {ffi_num}\n{N} Sample Predictions\n64x64 Model Average Prediction MSE Loss: {avg_pred_RMSE_Loss_64:.3e}\n128x128 Model Average Prediction MSE Loss: {avg_pred_RMSE_Loss_128:.3e}", fontsize = 25)

            # X
            img00 = axes[0,0].imshow(X, cmap='gray', vmin=0, vmax=1)
            axes[0,0].set_title(r"$X$" + "\nOriginal Image\n\n\n64x64 Model")
            axes[0,0].axis('off')
            fig.colorbar(img00, ax=axes[0,0], fraction=0.04)
            
            img10 = axes[1,0].imshow(X, cmap='gray', vmin=0, vmax=1)
            # axes[1,0].set_title(r"$X$" + "\nOriginal Image")
            axes[1,0].set_title("128x128 Model")
            axes[1,0].axis('off')
            fig.colorbar(img00, ax=axes[1,0], fraction=0.04)

            # X_proc
            img01 = axes[0,1].imshow(X_proc_64[0], cmap='gray', vmin=0, vmax=1)
            axes[0,1].set_title(r"$X_{proc}$" + "\nProcessed Image\n\n\n64x64 Model")
            axes[0,1].axis('off')
            fig.colorbar(img01, ax=axes[0,1], fraction=0.04)

            img11 = axes[1,1].imshow(X_proc_128[0], cmap='gray', vmin=0, vmax=1)
            # axes[1,1].set_title(r"$X_{proc}$" + "\nProcessed Image")
            axes[1,1].set_title("128x128 Model")
            axes[1,1].axis('off')
            fig.colorbar(img01, ax=axes[1,1], fraction=0.04)

            # X-X_proc
            img02 = axes[0,2].imshow(X-X_proc_upsampled_64[0], cmap='gray', vmin=0, vmax=1)
            axes[0,2].set_title(r"$X-X_{proc}$" + "\nProcessed Removed\nScattered Light\n\n64x64 Model")
            axes[0,2].axis('off')
            fig.colorbar(img02, ax=axes[0,2], fraction=0.04)

            img12 = axes[1,2].imshow(X-X_proc_upsampled_128[0], cmap='gray', vmin=0, vmax=1)
            # axes[1,2].set_title(r"$X-X_{proc}$" + "\nProcessed Removed\nScattered Light")
            axes[1,2].set_title("128x128 Model")
            axes[1,2].axis('off')
            fig.colorbar(img02, ax=axes[1,2], fraction=0.04)

            # bar(x)
            img03 = axes[0,3].imshow(bar_x_64[0], cmap='gray', vmin=0, vmax=1)
            axes[0,3].set_title(r"$\bar{x}$" + "\nMean Prediction\n\n\n64x64 Model")
            axes[0,3].axis('off')
            fig.colorbar(img03, ax=axes[0,3], fraction=0.04)

            img13 = axes[1,3].imshow(bar_x_128[0], cmap='gray', vmin=0, vmax=1)
            # axes[1,3].set_title(r"$\bar{x}$" + "\nMean Prediction")
            axes[1,3].set_title("128x128 Model")
            axes[1,3].axis('off')
            fig.colorbar(img03, ax=axes[1,3], fraction=0.04)
            
            # sigma_x
            img04 = axes[0,4].imshow(sigma_x_64[0], cmap='gray')
            axes[0,4].set_title(r"$\sigma_x$" + "\nStandard Deviation\nof Prediction\n\n64x64 Model")
            axes[0,4].axis('off')
            fig.colorbar(img04, ax=axes[0,4], fraction=0.04)

            img14 = axes[1,4].imshow(sigma_x_128[0], cmap='gray')
            # axes[1,4].set_title(r"$\sigma_x$" + "\nStandard Deviation\nof Prediction")
            axes[1,4].set_title("128x128 Model")
            axes[1,4].axis('off')
            fig.colorbar(img04, ax=axes[1,4], fraction=0.04)

            # X_proc - bar(x)
            img05 = axes[0,5].imshow(X_proc_upsampled_64[0]-bar_x_64[0], cmap='gray', vmin=0, vmax=1)
            axes[0,5].set_title(r"$X_{proc} - \bar{x}$" + "\nPrediction Error\n\n\n64x64 Model")
            axes[0,5].axis('off')
            fig.colorbar(img05, ax=axes[0,5], fraction=0.04)

            img15 = axes[1,5].imshow(X_proc_upsampled_128[0]-bar_x_128[0], cmap='gray', vmin=0, vmax=1)
            # axes[1,5].set_title(r"$X_{proc} - \bar{x}$" + "\nPrediction Error")
            axes[1,5].set_title("128x128 Model")
            axes[1,5].axis('off')
            fig.colorbar(img05, ax=axes[1,5], fraction=0.04)

            # X - bar(x)
            img06 = axes[0,6].imshow(X-bar_x_64[0], cmap='gray', vmin=0, vmax=1)
            axes[0,6].set_title(r"$X - \bar{x}$" + "\nMean Predicted Removed\nScattered Light\n\n64x64 Model")
            axes[0,6].axis('off')
            fig.colorbar(img06, ax=axes[0,6], fraction=0.04)

            img16 = axes[1,6].imshow(X-bar_x_128[0], cmap='gray', vmin=0, vmax=1)
            # axes[1,6].set_title(r"$X - \bar{x}$" + "\nMean Predicted Removed\nScattered Light")
            axes[1,6].set_title("128x128 Model")
            axes[1,6].axis('off')
            fig.colorbar(img06, ax=axes[1,6], fraction=0.04)
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
            # save images
            plt.tight_layout()
            fig.savefig(os.path.join(eval_folder_path, f"eval_{count}_{ffi_num}.pdf"))
            print('saved image at ' + os.path.join(eval_folder_path, f"eval_{count}_{ffi_num}.pdf"))
            plt.close()

            count += 1
            if count == 100: break

if __name__ == "__main__":
    display_TESS_two_models()

########## code for comparing two different models of 2 different sizes (end) ###############
'''


