# -*- coding: utf-8 -*-

from typing import Any, Literal

import numpy
import torch
import safetensors.torch

from seetapsych_lib import api

from .models.retinaface import RetinaFace
from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class Instance(api.Instance):
    def __init__(self, network: Literal['mobile0.25', 'resnet50'], model_path: str, device: api.Device,
                 threshold: float = 0.6):
        torch_device = torch.device(str(device))
        with torch.no_grad():
            match network:
                case 'mobile0.25':
                    cfg = cfg_mnet
                case 'resnet50':
                    cfg = cfg_re50
                case _:
                    raise RuntimeError(f'network must be mobile0.25 or resnet50, got {network}')
            cfg['pretrain'] = False

            net = RetinaFace(cfg=cfg, phase='test')

            state_dict = safetensors.torch.load_file(model_path)
            net.load_state_dict(state_dict, strict=False)
            net.eval()

            net.to(torch_device)

        self.__torch_device = torch_device
        self.__net = net
        self.__cfg = cfg
        self.__threshold = threshold

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        device = self.__torch_device
        net = self.__net
        cfg = self.__cfg
        output_threshold = self.__threshold

        confidence_threshold = 0.02
        nms_threshold = 0.4
        resize = 1
        top_k = 5000
        keep_top_k = 750

        with torch.no_grad():
            input_data = data['default']
            input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format

            img = numpy.float32(input_data)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            loc, conf, landms = net(img)

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

        # ignore low scores
        inds = numpy.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = numpy.hstack((boxes, scores[:, numpy.newaxis])).astype(numpy.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = numpy.concatenate((dets, landms), axis=1)
        # [N, [x, y, x, y, score, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y]]
        dets = dets[dets[:, 4] >= output_threshold]

        face_detection = []
        face_landmarks = []

        for det in dets:
            det = det.tolist()
            face_detection.append({
                'xyxy': det[:4],
                'score': det[4]
            })
            face_landmarks.append({
                'landmarks': det[5:]
            })

        return {
            'face_detection': face_detection,
            'face_landmarks': face_landmarks,
        }


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        threshold = parameters.get('threshold', 0.6)

        model_path = models[0].cache()
        return Instance(
            models[0].metadata['network'],
            model_path,
            api.Device('cpu') if device is None else device,
            threshold=threshold,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
