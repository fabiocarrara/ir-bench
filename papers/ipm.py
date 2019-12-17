import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('Paired')[3:6]  # reuse colors of other plot

sns.set(style='whitegrid', context='paper', palette=palette, font='serif', font_scale=1.4, color_codes=False, rc={
    'text.usetex': True,
    'legend.frameon': True,
})

import pandas as pd

from expman import Experiment

def effectiveness_vs_timespace(args):
    
    db_sizes = [50000, 100000, 250000, 500000, 750000, 950000]

    def _faiss_gather(s_dir):
        
        faiss_exps = os.path.join(args.run, s_dir)
        faiss_exps = Experiment.gather(faiss_exps)
        faiss_exps = Experiment.filter(args.filter, faiss_exps)
        faiss_exps = list(faiss_exps)

        db_sizes = [50000, 100000, 250000, 500000, 750000, 950000]
        effec = Experiment.collect_all(faiss_exps, 'metrics*.csv')
        effec = effec.query('limit in @db_sizes')

        times = Experiment.collect_all(faiss_exps, 'query_times.csv')
        times['query_time'] = times.loc[:, 'query_time_run1':'query_time_run5'].mean(axis=1)

        space = Experiment.collect_all(faiss_exps, 'index_stats.csv')
        # space['size'] *= 64 / 1024 
        space['size'] /= 1024 ** 2
        space['build_time'] = space.train_time + space.add_time
        
        data = effec.merge(times)
        data = data.merge(space)
        data['limit'] = data.limit.apply(lambda x: str(x)[:-3] + 'k')

        data = pd.pivot_table(data, values=['ap', 'ndcg', 'ndcg@25', 'query_time', 'size', 'build_time'], index=['limit', 'n_probes'])
        data = data.reset_index().rename(columns={'limit': 'samples', 'n_probes': 'trade-off'})
        print(data)
        return data
   
    # FAISS
    fhdata = _faiss_gather('ivfpq_H+M100k')
    fhdata['method'] = 'IVFPQ'
    # FAISS (T4SA)
    ftdata = _faiss_gather('ivfpq_bt4sa')
    ftdata['method'] = 'IVFPQ*'
    
    # THR SQ LUCENE
    thr_exps = os.path.join(args.run, 'lucene-thr-sq')
    thr_exps = Experiment.gather(thr_exps)
    thr_exps = Experiment.filter(args.filter, thr_exps)
    thr_exps = list(thr_exps)

    effec = Experiment.collect_all(thr_exps, 'metrics.csv')
    effec = effec.query('limit in @db_sizes')

    space = Experiment.collect_all(thr_exps, 'index_stats.csv')
    space['size'] /= 10 ** 6
    data = effec.merge(space)
    data['limit'] = data.limit.apply(lambda x: str(x)[:-3] + 'k')
    data['build_time'] = data.add_time
    
    data = pd.pivot_table(data, values=['ap', 'ndcg', 'ndcg@25', 'query_time', 'size', 'build_time'], index=['limit', 'threshold'])
    sqdata = data.reset_index().rename(columns={'limit': 'samples', 'threshold': 'trade-off'})
    print(sqdata)
    sqdata['method'] = 'Thr-SQ'

    style_order = [str(x)[:-3] + 'k' for x in db_sizes]

    data = pd.concat((sqdata, ftdata, fhdata))

    for metric in args.metrics:
        # EvT PLOT
        plt.figure()
        ax = sns.lineplot(x='query_time', y=metric, hue='method', style='samples', markers=True, style_order=style_order, data=data)
        '''
        ax = plt.gca()
        common = dict(x='query_time', y=metric, style='samples', markers=True, style_order=style_order, ax=ax)
        sns.lineplot(data=sqdata, **common)  # size='threshold'
        sns.lineplot(data=ftdata, **common)  # size='n_probes'
        sns.lineplot(data=fhdata, **common)  # size='n_probes'
        '''
        ax.set(xscale='log')
        # plt.legend(title='DB size $N$')
        plt.xlabel('Query Time (s)')
        plt.ylabel('mAP' if metric == 'ap' else metric.upper())
        out = f'evt_{metric}.pdf'
        plt.savefig(out)
        os.system(f'croppdf {out}')
        plt.close()

        # EvS PLOT
        plt.figure()
        ax = sns.lineplot(x='size', y=metric, hue='method', style='samples', markers=True, style_order=style_order, estimator=None, data=data)
        '''
        ax = plt.gca()
        common = dict(x='size', y=metric, style='samples', markers=True, style_order=style_order, ax=ax, estimator=None)
        sns.lineplot(data=sqdata, **common)  # size='threshold'
        sns.lineplot(data=ftdata, **common)  # size='n_probes'
        sns.lineplot(data=fhdata, **common)  # size='n_probes'
        '''
        ax.set(xscale='log')
        plt.xlabel('Index Size (MB)')
        plt.ylabel('mAP' if metric == 'ap' else metric.upper())
        out = f'evs_{metric}.pdf'
        plt.savefig(out)
        os.system(f'croppdf {out}')
        plt.close()

        # EvB PLOT
        plt.figure()
        ax = sns.lineplot(x='build_time', y=metric, hue='method', style='samples', markers=True, style_order=style_order, estimator=None, data=data)
        '''
        ax = plt.gca()
        common = dict(x='size', y=metric, style='samples', markers=True, style_order=style_order, ax=ax, estimator=None)
        sns.lineplot(data=sqdata, **common)  # size='threshold'
        sns.lineplot(data=ftdata, **common)  # size='n_probes'
        sns.lineplot(data=fhdata, **common)  # size='n_probes'
        '''
        # ax.set(xscale='log')
        plt.xlabel('Indexing Time (s)')
        plt.ylabel('mAP' if metric == 'ap' else metric.upper())
        out = f'evb_{metric}.pdf'
        plt.savefig(out)
        os.system(f'croppdf {out}')
        plt.close()


if __name__ == '__main__':

    def run_filter(string):
        if '=' not in string:
            raise argparse.ArgumentTypeError(
                f'Filter {string} is not in format <param1>=<value1>[, <param2>=<value2>[, ...]]')
        filters = string.split(',')
        filters = map(lambda x: x.split('='), filters)
        filters = {k: v for k, v in filters}
        return filters


    parser = argparse.ArgumentParser(description='Plot stuff')
    parser.add_argument('-f', '--filter', default={}, type=run_filter)
    subparsers = parser.add_subparsers()

    parser_evts = subparsers.add_parser('effectiveness_vs_timespace')
    parser_evts.add_argument('run', default='runs/')
    parser_evts.add_argument('-m', '--metrics', nargs='+', default=('ap', 'ndcg', 'ndcg@25'))
    parser_evts.set_defaults(func=effectiveness_vs_timespace)

    args = parser.parse_args()
    args.func(args)
