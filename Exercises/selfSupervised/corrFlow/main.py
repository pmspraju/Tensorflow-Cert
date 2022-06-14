import argparse
import os, time
import numpy as np

import selfSupervised.corrFlow.functional.feeder.dataset.KineticsFull as K
import selfSupervised.corrFlow.functional.feeder.dataset.KineticsLoader as KL

#from models.corrflow import CorrFlow
#from test import test

import selfSupervised.corrFlow.logger as logger

def main():

    #log = logger.setup_logger(args.savepath + '/training.log')
    datapath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\Kinetics'
    jsonpath = os.path.join(datapath, 'kinetics400')
    jsonpath = os.path.join(jsonpath, 'train.json')
    #TrainData = K.dataloader(csvpath)
    KL.read_videos(datapath, jsonpath)

    # TrainImgLoader = torch.utils.data.DataLoader(
    #     KL.myImageFloder(args.datapath, TrainData, True),
    #     batch_size=args.bsize, shuffle=True, num_workers=args.worker,drop_last=True
    # )
    #
    # model = CorrFlow(args)
    # model = nn.DataParallel(model).cuda()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    #
    # log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    #
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         log.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         log.info("=> loaded checkpoint '{}'".format(args.resume))
    #     else:
    #         log.info("=> No checkpoint found at '{}'".format(args.resume))
    #         log.info("=> Will start from scratch.")
    # else:
    #     log.info('=> No checkpoint file. Start from scratch.')
    #
    # start_full_time = time.time()
    #
    # for epoch in range(args.epochs):
    #     log.info('This is {}-th epoch'.format(epoch))
    #     train(TrainImgLoader, model, optimizer, log, epoch)
    #
    # log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()