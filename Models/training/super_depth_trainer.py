
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.super_depth_network import SuperDepthNetwork
from data_utils.augmentations import Augmentations

class SuperDepthTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', size=''):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.gt_fused = 0
        self.gt_val_fused = 0
        self.augmented = 0
        self.augmented_val = 0
        self.image_tensor = 0
        self.image_val_tensor = 0
        self.gt_tensor = 0
        self.gt_val_tensor = 0
        self.loss = 0
        self.prediction = 0
        self.calc_loss = 0
        self.prediction_vis = 0
        self.checkpoint_path = checkpoint_path

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
        
        # Instantiate Model with pre-trained weights
        sceneSegNetwork = SceneSegNetwork()
        if(len(pretrained_checkpoint_path) > 0):
            sceneSegNetwork.load_state_dict(torch.load \
                (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No pre-trained model checkpoint path - pass in pretrained SceneSeg checkpoint path')
        
        self.model = SuperDepthNetwork(sceneSegNetwork)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    # Logging Training Loss
    def log_loss(self, log_count):
        print('Logging Training Loss', log_count, self.get_loss())
        self.writer.add_scalar("Loss/train", self.get_loss(), (log_count))

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    def set_val_data(self, image_val, gt_val):
        self.image_val = image_val
        self.gt_val = gt_val
    '''
    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Image
            aug_train = Augmentations(self.image, self.gt, True)
            self.image, self.augmented = aug_train.getAugmentedData()
        else:
            # Augmenting Image
            aug_val = Augmentations(self.image_val, self.gt_val, False)
            self.image_val, self.augmented_val = aug_val.getAugmentedData()
    '''
    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)


    # Run Model
    def run_model(self):     
        self.loss = nn.L1Loss()
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    # Loss Backward Pass
    def loss_backward(self): 
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()
    '''
    # Save predicted visualization
    def save_visualization(self, log_count):
        print('Saving Visualization')
        self.prediction_vis = self.prediction.squeeze(0).cpu().detach()
        self.prediction_vis = self.prediction_vis.permute(1, 2, 0)
                
        vis_predict = self.make_visualization()
        label = self.augmented[0]
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(label)
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[2].imshow(vis_predict)
        axs[2].set_title('Prediction',fontweight ="bold") 
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))
    '''
    # Load Image as Tensor
    def load_image_tensor(self, is_train):

        if(is_train):
            image_tensor = self.image_loader(self.image)
            image_tensor = image_tensor.unsqueeze(0)
            self.image_tensor = image_tensor.to(self.device)
        else:
            image_val_tensor = self.image_loader(self.image_val)
            image_val_tensor = image_val_tensor.unsqueeze(0)
            self.image_val_tensor = image_val_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self, is_train):

        if(is_train):
            gt_tensor = torch.from_numpy(self.gt)
            gt_tensor = torch.div(gt_tensor, 7)
            gt_tensor = gt_tensor.unsqueeze(0)
            self.gt_tensor = gt_tensor.to(self.device)
        else:
            gt_val_tensor = torch.from_numpy(self.gt_val)
            gt_val_tensor = torch.div(gt_val_tensor, 7)
            gt_val_tensor = gt_val_tensor.unsqueeze(0)
            self.gt_val_tensor = gt_val_tensor.to(self.device)
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)
    
    '''
    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val):

        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Calculate IoU score
        iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd \
            = self.calc_IoU_val()
        
        return iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd


    # Visualize predicted result
    def make_visualization(self):
        shape = self.prediction_vis.shape
        _, output = torch.max(self.prediction_vis, dim=2)

        row = shape[0]
        col = shape[1]
        vis_predict = Image.new(mode="RGB", size=(col, row))
    
        vx = vis_predict.load()

        background_objects_colour = (61, 93, 255)
        foreground_objects_colour = (255, 28, 145)
        road_colour = (0, 255, 220)

        # Extracting predicted classes and assigning to colourmap
        for x in range(row):
            for y in range(col):
                if(output[x,y].item() == 0):
                    vx[y,x] = background_objects_colour
                elif(output[x,y].item() == 1):
                    vx[y,x] = foreground_objects_colour
                elif(output[x,y].item() == 2):
                    vx[y,x] = road_colour               
        
        return vis_predict
    '''
    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')