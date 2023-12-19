from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from model.pytorch.supervisor import SGODESupervisor 
from lib.utils import load_graph_data

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        save_adj_name = args.config_filename[11:-5]
        supervisor = SGODESupervisor(args,save_adj_name, temperature=args.temperature, **supervisor_config)
        if args.mode == 'train':
            supervisor.train()
        elif args.mode == 'eval':
            supervisor.pre(args.pre_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/para_la.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
    parser.add_argument('--model',type=str,default='SGODE-RNN')
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--pre_model',type=str,default='None')
    args = parser.parse_args()
    main(args)