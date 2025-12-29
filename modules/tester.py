import logging
from abc import abstractmethod

# import scispacy
import torch
from datetime import datetime


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # set hadler 
        dt = datetime.strftime(datetime.now(), "%Y-%m-%d_%H")
        experiment_type = "experiment"
        logname = "logfile_saved/test_{}_{}_{}.log".format(args.dataset_name,str(dt),experiment_type)
        file_handler = logging.FileHandler(logname, 'w')
        self.logger.addHandler(file_handler)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader
        self.test = self.test_IU_Xray

    def test_IU_Xray(self):
        self.logger.info('Start to evaluate in the IU Xray test set.')
        self.model.eval()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            # 新增，用于保存预测报告的文件，以追加模式打开，可根据实际情况修改文件名和路径
            prediction_report_file = open('logfile_saved/prediction_report.txt', 'a', encoding='utf-8')
            for _, (images_id,
                        images,
                        image_mask_bone,
                        image_mask_lung,
                        image_mask_heart,
                        image_mask_mediastinum,
                        reports_ids,
                        reports_masks,
                        labels,
                        disease_detected) in enumerate(self.test_dataloader):
                
                images = images.to(self.device)
                image_mask_bone = image_mask_bone.to(self.device)
                image_mask_lung = image_mask_lung.to(self.device)
                image_mask_heart = image_mask_heart.to(self.device)
                image_mask_mediastinum = image_mask_mediastinum.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                output, _ = self.model(images, 
                                       image_mask_bone,
                                       image_mask_lung, 
                                       image_mask_heart, 
                                       image_mask_mediastinum, 
                                       mode='sample')
                
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                # 将每一个批次的预测报告逐行写入文件
                for pred_report in reports:
                    prediction_report_file.write(pred_report + '\n')

            prediction_report_file.close()  # 关闭文件

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)
            for idx in range(len(test_gts)): #random.sample(range(len(test_gts)), 10):
                self.logger.info(">>>> The example idx is {}".format(idx))
                self.logger.info(">>>> test Example predict: {}.".format(test_res[idx]))
                self.logger.info(">>>> test Example target : {}.".format(test_gts[idx]))
                
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))
        return log
    
