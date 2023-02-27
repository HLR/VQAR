import os
import cProfile
import pstats
import io
import sys
from pstats import SortKey

supervised_learning_path = os.path.abspath(os.path.join(
    os.path.abspath(__file__), "../supervised_learning"))
sys.path.insert(0, supervised_learning_path)

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

proj_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))

print(proj_path)

from cmd_args import cmd_args
from trainer import ClauseNTrainer
from utils import gen_task
import pickle
from dataloader import ListDataLoader

def train(args):

    print(f"training file: {args.train_f}")
    print(f"validating file: {args.val_f}")
    print(f"models are saved to: {args.model_dir}")

    with open (args.train_f, 'rb') as train_task_file:
        train_tasks = pickle.load(train_task_file)
    with open (args.val_f, 'rb') as val_task_file:
        val_tasks = pickle.load(val_task_file)
    # val_tasks = []

    train_data_loader = ListDataLoader(train_tasks)
    val_data_loader = ListDataLoader(val_tasks)

    clause_n_trainer = ClauseNTrainer(train_data_loader, val_data_loader)
    clause_n_trainer.train()

def test(args):

    print(f"test on : {args.test_f}")
    with open (args.test_f, 'rb') as test_task_file:
        test_tasks = pickle.load(test_task_file)

    test_data_loader = ListDataLoader(test_tasks)
    clause_n_trainer = ClauseNTrainer(train_data_loader=None, val_data_loader=None, test_data_loader=test_data_loader)
    clause_n_trainer.test()

if __name__ == "__main__":

    profile = False
    if profile:
        data_dir = os.path.abspath(os.path.join(
            os.path.abspath(__file__), "../../../data"))
        strange_case_dir = os.path.join(data_dir, "debug")
        stats_filename = os.path.join(strange_case_dir, "stats.log")

        pr = cProfile.Profile()
        pr.enable()

    print(f'REINFORCE is {cmd_args.reinforce}')
    # relevant args
    print('args')
    print(f'replays: {cmd_args.replays}')
    print(f'train_f: {cmd_args.train_f}')
    print(f'val_f: {cmd_args.val_f}')
    print(f'test_f: {cmd_args.test_f}')
    print(f'model_dir: {cmd_args.model_dir}')

    train(cmd_args)
    test(cmd_args)

    if profile:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()

        with open(stats_filename, 'w') as stats_file:
            stats_file.write(stats)

    print('end')
