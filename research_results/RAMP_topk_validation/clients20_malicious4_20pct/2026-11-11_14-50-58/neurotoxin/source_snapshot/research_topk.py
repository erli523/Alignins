#!/usr/bin/env python3
"""Isolated Top-K research entry; execute the unmodified Alignins training code."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()
        return len(message)
    def flush(self):
        self.stream.flush()
        if not self.file.closed:
            self.file.flush()
    def isatty(self):
        return False


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                         encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, help='New, non-existing experiment directory')
    parser.add_argument('--topk-ratio', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=0.001)
    parser.add_argument('--reference', choices=['median', 'mean'], default='median')
    parser.add_argument('--l2-lower', type=float, default=25)
    parser.add_argument('--l2-upper', type=float, default=75)
    parser.add_argument('--min-reference', type=int, default=8)
    parser.add_argument('--save-indices', choices=['selected', 'all', 'none'], default='selected')
    parser.add_argument('--snapshot-every', type=int, default=25)
    parser.add_argument('--report-only', type=Path)
    parser.add_argument('--compare', type=Path)
    options, native_argv = parser.parse_known_args()
    if options.report_only or options.compare:
        if native_argv:
            parser.error('Unexpected training arguments in report-only mode')
        from research.topk_report import make_report, compare_batch
        if options.report_only:
            make_report(options.report_only.resolve())
        else:
            compare_batch(options.compare.resolve())
        return
    if not options.output:
        parser.error('--output is required for training')
    if (not math.isfinite(options.topk_ratio) or not 0 < options.topk_ratio <= 1
            or not math.isfinite(options.epsilon) or options.epsilon <= 0
            or not 0 <= options.l2_lower < options.l2_upper <= 100
            or options.min_reference < 1 or options.snapshot_every < 1):
        parser.error('Invalid diagnostic parameters')
    output = options.output.resolve()
    # Refuse to reuse a directory so a rerun cannot silently overwrite previous results.
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='initializing', started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                    native_argv=native_argv, diagnostics=vars(options),
                    native_seed=0, formula='abs(local_delta)/(abs(reference)+epsilon)',
                    scope='requires_grad model parameters; observe before native filtering',
                    median_convention='torch.median: lower middle value for even sample counts',
                    ground_truth_usage='role labels for evaluation only; no role-based reference selection',
                    source_sha256={})
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for source in [ROOT / 'research_topk.py', ROOT / 'run_ramp_topk.sh',
                   ROOT / 'research' / 'topk_diagnostics.py', ROOT / 'research' / 'topk_report.py',
                   *sorted((ROOT / 'src').glob('*.py'))]:
        if source.exists():
            relative = source.relative_to(ROOT)
            metadata['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    save_json(output / 'run.json', metadata)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    console = (output / 'console.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(original_stdout, console), Tee(original_stderr, console)
    observer = metrics = None
    started = time.perf_counter()
    failure = None
    try:
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT / 'src'))
        import utils
        from aggregation import Aggregation
        from research.topk_diagnostics import TopKObserver, MetricsHandler
        observer = TopKObserver(output, options)
        metrics = MetricsHandler(output)
        metadata.update(torch_version=torch.__version__, numpy_version=np.__version__,
                        cuda_runtime=torch.version.cuda)
        original_aggregate = Aggregation.aggregate_updates

        def setup_research_logging(args):
            args.exp_name = 'RAMP_topk_diagnostic'
            metadata['training'] = vars(args).copy()
            metadata['status'] = 'running'
            if str(args.device).startswith('cuda'):
                metadata['gpu_name'] = torch.cuda.get_device_name(args.device)
            save_json(output / 'run.json', metadata)
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
                handler.close()
            root_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            root_logger.addHandler(handler)
            root_logger.addHandler(metrics)
            logging.info('Independent RAMP Top-K experiment: %s', output)
            logging.info('Diagnostic formula: %s; native aggregation is unchanged', metadata['formula'])
            logging.info('Training arguments: %s', args)
            return str(output)

        def observe_then_aggregate(aggregator, model, updates):
            observer.observe(model, updates, aggregator.args)
            if observer.round == 1:
                metadata['parameter_layout'] = observer.layout
                metadata['analysis_parameter_count'] = observer.dimension
                metadata['uploaded_state_count'] = observer.full_dimension
                save_json(output / 'run.json', metadata)
            return original_aggregate(aggregator, model, updates)

        # Patches live only inside this dedicated process. Source files remain untouched.
        utils.setup_logging = setup_research_logging
        Aggregation.aggregate_updates = observe_then_aggregate
        os.chdir(ROOT / 'src')
        sys.argv = [str(ROOT / 'src' / 'federated.py'), *native_argv]
        runpy.run_path(str(ROOT / 'src' / 'federated.py'), run_name='__main__')
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        failure = exc
    finally:
        if observer:
            observer.close()
        if metrics:
            metrics.close()
        metadata.update(finished_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                        elapsed_seconds=time.perf_counter() - started,
                        observed_rounds=observer.round if observer else 0)
        save_json(output / 'run.json', metadata)
        try:
            from research.topk_report import make_report
            make_report(output)
            metadata['report_status'] = 'generated'
        except Exception as exc:
            metadata['report_status'] = f'failed: {type(exc).__name__}: {exc}'
            traceback.print_exc()
            if failure is None:
                failure = exc
        save_json(output / 'run.json', metadata)
        print(f'Experiment status: {metadata["status"]}; results: {output}', flush=True)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        console.close()
    if failure is not None:
        raise SystemExit(130 if isinstance(failure, KeyboardInterrupt) else 1)


if __name__ == '__main__':
    main()
