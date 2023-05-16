# from mimetypes import init
# from time import time
# from turtle import end_fill
from data.tracking.post_processor.response_map import ResponseMapTrackingPostProcessing
from runners.interface import BaseRunner
from data.tracking.methods.sequential.curation_parameter_provider import SiamFCCurationParameterSimpleProvider
from data.tracking.methods.SiamFC.common.siamfc_curation import do_SiamFC_curation
import torch
from data.tracking.post_processor.bounding_box.default import DefaultBoundingBoxPostProcessor 
from data.operator.bbox.spatial.vectorized.torch.utility.normalize import BoundingBoxNormalizationHelper
from data.types.bounding_box_format import BoundingBoxFormat
import time
import config.global_var as gv

def _run_fn(fn, args):
    if isinstance(args, (list, tuple)):
        return fn(*args)
    elif isinstance(args, dict):
        return fn(**args)
    else:
        return fn(args)

class Tracker:
    def __init__(self,runner,branch_name):
        self.device = runner.tracker_evaluator[branch_name].device
        self.search_image_curation_parameter_provider  = runner.tracker_evaluator[branch_name].search_curation_parameter_provider
        self.search_curation_image_size = runner.tracker_evaluator[branch_name].search_curation_image_size
        self.bounding_box_post_processor = runner.tracker_evaluator[branch_name].bounding_box_post_processor
        self.post_processor = runner.tracker_evaluator[branch_name].post_processor
        self.interpolation_mode = runner.tracker_evaluator[branch_name].interpolation_mode
        self.template_curated_image_shape = runner.tracker_evaluator[branch_name].template_curated_image_cache_shape
    
    def initialize_tracker(self,video_data):
        self.z_curated = video_data['z_curated']
        self.z_curated = self.z_curated.to(device = self.device)
        self.z_bbox = video_data['z_bbox']
        self.z_image_mean = video_data['z_image_mean']
        self.full_image = video_data['x']
        self.frame_index = video_data['frame_index']
        self.z_feat = video_data['z_feat']
        search_image = self.full_image.to(device=self.device)
        search_image_size = search_image.shape[1:]
        self.search_image_size = torch.tensor((search_image_size[1], search_image_size[0]),device=self.device)  # (W, H)
        self.predicted_iou = None

        
    def run_tracking( self,model):
        search_image = self.full_image.to(device=self.device)

        # get template feature from 1st frame
        if self.frame_index == 1 :
            self.search_image_curation_parameter_provider.initialize(self.z_bbox) 
            self.z_feat  = _run_fn(model.initialize, self.z_curated.unsqueeze(0))
            
        if self.update_template and self.predicted_iou>gv.iou_threshold and gv.trident:
            z_feat_new = _run_fn(model.initialize,self.new_template.unsqueeze(0).to(device = self.device))
            self.z_feat,_ = model.concatenation(self.z_feat,z_feat_new,z_feat_new)
            self.update_template= False

        
        curation_parameter = self.search_image_curation_parameter_provider.get(self.search_curation_image_size)
        search_curated_image,_ = do_SiamFC_curation(search_image, self.search_curation_image_size, curation_parameter,
                                self.interpolation_mode, self.z_image_mean)
        tracking_sample = {
            'z_feat' : self.z_feat,
            'x' : search_curated_image.unsqueeze(0).to(device=self.device)
            }
            
        #get output of model
        output = None
        if tracking_sample is not None:
            output = _run_fn(model.track, tracking_sample)
            output = self.post_processor(output)
            self.predicted_iou, predicted_bounding_box = output['conf'], output['bbox']
            predicted_bounding_box = predicted_bounding_box.to(torch.float64)
            curation_parameter = curation_parameter.unsqueeze(0).to(device=self.device)
            
            #convert output to pixel
            predicted_bounding_box = self.bounding_box_post_processor(predicted_bounding_box, curation_parameter[:predicted_bounding_box.shape[0], ...])

        #update curation parameter for search image for next frame

        # don't update search window size and position until iou > threshold
        if self.predicted_iou > 0.89:
            self.search_image_curation_parameter_provider.update(self.predicted_iou, predicted_bounding_box.squeeze(0), self.search_image_size)
        return predicted_bounding_box.squeeze(0),self.predicted_iou