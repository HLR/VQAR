import os
import cProfile
import pstats
import io
import sys
from pstats import SortKey


common_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.insert(0, common_path)

from cmd_args import cmd_args
from trainer import ClauseNTrainer
from utils import gen_task
import pickle
from dataloader import ListDataLoader


def test(args, test_path):

    if test_path is not None:
        test_path = cmd_args.test_f

    with open (test_path, 'rb') as test_task_file:
        test_data_generator = pickle.load(test_task_file)

    test_data_generator = ListDataLoader(test_data_generator)
    clause_n_trainer = ClauseNTrainer(train_data_loader=None, val_data_loader=None, test_data_loader=test_data_generator)
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

    test(cmd_args, test_path=cmd_args.test_f)

    if profile:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue()

    print('end')

