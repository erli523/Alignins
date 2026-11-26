"""CPU regression tests; no dataset download and no modification of native files."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import algorithms as defenses
from .__main__ import parse_args


class EquationTests(unittest.TestCase):
    def test_fedavg_sample_weights(self):
        result = defenses.fedavg(torch.tensor([1., 2.]), {2: torch.tensor([2., 4.]), 9: torch.tensor([6., 8.])}, {2: 3, 9: 1}, 0.5)
        torch.testing.assert_close(result.model, torch.tensor([2.5, 4.5]))

    def test_rlr_boundary_and_unweighted_votes(self):
        updates = {0: torch.tensor([1., -2., 0.]), 4: torch.tensor([3., 4., 0.]), 8: torch.tensor([-2., 6., 0.])}
        sizes = {0: 100, 4: 1, 8: 1}
        result = defenses.rlr(torch.zeros(3), updates, sizes, threshold=1, server_lr=0.5)
        expected = (100 * updates[0] + updates[4] + updates[8]) / 102 * torch.tensor([.5, .5, -.5])
        torch.testing.assert_close(result.model, expected)
        self.assertEqual(result.diagnostics["flipped_coordinates"], 1)
        torch.testing.assert_close(defenses.rlr(torch.ones(3), updates, sizes, 0).model,
                                   defenses.fedavg(torch.ones(3), updates, sizes).model)

    def test_flame_local_model_cosines_and_even_median(self):
        base = torch.tensor([10., 2.])
        updates = {0: torch.tensor([1., 0.]), 1: torch.tensor([3., 0.]), 2: torch.tensor([5., 0.]), 3: torch.tensor([-20., 0.])}
        with patch("hdbscan.HDBSCAN") as clustering:
            clustering.return_value.fit_predict.return_value = np.array([0, 0, 0, -1])
            result = defenses.flame(base, updates, 0)
            local = torch.stack([base + value for value in updates.values()]).double()
            local /= local.norm(dim=1, keepdim=True)
            expected = (1 - local @ local.T).clamp(0, 2).numpy()
            np.fill_diagonal(expected, 0)
            np.testing.assert_allclose(clustering.return_value.fit_predict.call_args.args[0], expected)
            self.assertEqual(clustering.call_args.kwargs["min_samples"], 1)
            self.assertEqual(clustering.call_args.kwargs["min_cluster_size"], 3)
        self.assertEqual(result.diagnostics["clip_bound"], 4)
        torch.testing.assert_close(result.model, base + torch.tensor([8 / 3, 0.]))

    def test_flame_no_cluster_does_not_fallback(self):
        with patch("hdbscan.HDBSCAN") as clustering:
            clustering.return_value.fit_predict.return_value = np.array([-1, -1, -1])
            with self.assertRaises(RuntimeError):
                defenses.flame(torch.ones(2), {client: torch.ones(2) for client in range(3)})

    def test_flame_actual_hdbscan(self):
        base = torch.tensor([10., 1., 1.])
        updates = {0: torch.tensor([.1, 0., 0.]), 1: torch.tensor([.11, .001, 0.]),
                   2: torch.tensor([.12, -.001, 0.]), 3: torch.tensor([-20., -10., 20.]),
                   4: torch.tensor([-30., 10., -20.])}
        result = defenses.flame(base, updates, 0)
        self.assertEqual(result.diagnostics["retained_clients"], [0, 1, 2])

    def test_flame_seed_reproducibility(self):
        updates = {client: torch.ones(4) for client in range(4)}
        with patch("hdbscan.HDBSCAN") as clustering:
            clustering.return_value.fit_predict.return_value = np.zeros(4, dtype=int)
            first = defenses.flame(torch.ones(4), updates, generator=torch.Generator().manual_seed(5))
            second = defenses.flame(torch.ones(4), updates, generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(first.model, second.model)

    def test_crfl_clips_global_model_not_update_and_final_no_noise(self):
        result = defenses.crfl(torch.tensor([3., 4.]), {1: torch.zeros(2)}, {1: 1}, 2, 100, True)
        torch.testing.assert_close(result.model, torch.tensor([1.2, 1.6]))
        self.assertEqual(result.diagnostics["training_noise_std"], 0)

    def test_crfl_noise_follows_clipping(self):
        generator = torch.Generator().manual_seed(21)
        expected = torch.tensor([1.2, 1.6]) + torch.randn(2, generator=torch.Generator().manual_seed(21)) * .1
        result = defenses.crfl(torch.tensor([3., 4.]), {0: torch.zeros(2)}, {0: 1}, 2, .1, False, generator)
        torch.testing.assert_close(result.model, expected)
        torch.testing.assert_close(result.evaluation_model, torch.tensor([1.2, 1.6]))

    def test_prodigal_strict_votes_uniform_mean_and_personalization(self):
        local = {0: torch.tensor([2., 20., 200., 2.]), 1: torch.tensor([4., 40., 400., 4.]), 2: torch.tensor([6., 60., 600., 6.])}
        initial = {client: torch.ones(4) for client in local}
        masks = {0: torch.tensor([True, True, False, False]), 1: torch.tensor([True, False, True, False]),
                 2: torch.tensor([True, False, False, False])}
        result = defenses.prodigal(local, initial, masks, 1)
        torch.testing.assert_close(result.model, torch.tensor([4., 0., 0., 0.]))
        torch.testing.assert_close(result.personalized[0], torch.tensor([4., 20., 0., 0.]))
        torch.testing.assert_close(result.personalized[1], torch.tensor([4., 0., 400., 0.]))
        torch.testing.assert_close(result.feedback, torch.tensor([3., 39., 399., 3.]))
        following = defenses.prodigal(result.personalized, result.personalized, masks, 1)
        torch.testing.assert_close(following.personalized[0], result.personalized[0])
        torch.testing.assert_close(following.feedback, torch.zeros(4))

    def test_prodigal_topk_is_final_gradient_and_per_layer_floor(self):
        model = torch.nn.Linear(4, 1)
        model.weight.grad = torch.tensor([[1., 9., 8., 2.]])
        model.bias.grad = torch.tensor([100.])
        layout = defenses.ParameterLayout(model)
        torch.testing.assert_close(layout.topk_gradients(model, .5), torch.tensor([False, True, True, False, False]))

    def test_layout_rejects_buffers_and_invalid_vectors(self):
        with self.assertRaises(ValueError):
            defenses.ParameterLayout(torch.nn.BatchNorm1d(2))
        model = torch.nn.Linear(2, 1, bias=False)
        with self.assertRaises(ValueError):
            defenses.ParameterLayout(model).load(model, torch.tensor([float("nan"), 0.]))

    def test_lockdown_consensus_does_not_modify_training_model(self):
        model = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        masks = [{"weight": torch.tensor([[1., 1., 0.]])}, {"weight": torch.tensor([[1., 0., 1.]])}]
        clean = defenses.consensus_model(model, masks, 2)
        torch.testing.assert_close(clean.weight, torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(model.weight, torch.ones(1, 3))

    def test_smoothed_votes_preserve_model_rng_and_noise_bank(self):
        model = torch.nn.Linear(3, 2)
        saved = copy.deepcopy(model.state_dict())
        inputs = torch.tensor([[1., 2., 3.], [2., 1., 0.]])
        smoothing = defenses.SmoothedClassifier(model, .2, 5, 8)
        rng = torch.get_rng_state().clone()
        whole = smoothing(inputs)
        split = torch.cat([smoothing(inputs[:1]), smoothing(inputs[1:])])
        torch.testing.assert_close(whole, split)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.testing.assert_close(whole.sum(1), torch.ones(2))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name])

    def test_invalid_update_and_sample_weight_fail(self):
        with self.assertRaises(ValueError):
            defenses.fedavg(torch.zeros(2), {0: torch.ones(2)}, {0: 0})
        with self.assertRaises(ValueError):
            defenses.rlr(torch.zeros(2), {0: torch.tensor([float("inf"), 0.])}, {0: 1}, 1)


class InterfaceTests(unittest.TestCase):
    def test_unverified_methods_fail_closed(self):
        for method in ["apr_fl", "fedtof"]:
            with self.subTest(method=method), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--method", method, "--dry-run"])

    def test_crfl_requires_explicit_cifar_hyperparameters(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--method", "crfl", "--dry-run"])

    def test_lockdown_unsupported_attack_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--method", "lockdown", "--attack", "pgd", "--dry-run"])

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new"
            args = parse_args(["--method", "rlr", "--dry-run", "--output", str(path)])
            self.assertFalse(args.output.exists())


class FakeCifar(torch.utils.data.Dataset):
    def __init__(self, count=80):
        self.data = np.random.default_rng(7).integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8)
        self.targets = torch.arange(count) % 10

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor(self.data[index].copy()).permute(2, 0, 1).float() / 255, self.targets[index]


class RunnerTests(unittest.TestCase):
    def test_native_pgd_projection_branch_and_no_module_patch(self):
        from . import runner
        helpers = runner.native_helpers()
        original_utility = helpers.core.utils.vector_to_model_wo_load
        original_train = helpers.Agent.local_train
        settings = parse_args(["--method", "fedavg", "--attack", "pgd", "--dry-run",
                               "--num-agents", "4", "--num-corrupt", "1", "--bs", "2",
                               "--local-ep", "1", "--max-batches", "1", "--client-lr", "100"])
        args = runner.native_args(settings, helpers)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 10))
        layout = defenses.ParameterLayout(model)
        before = layout.vector(model)
        agents, sizes = runner.make_agents(settings, args, helpers, FakeCifar(),
                                           {client: list(range(client * 20, (client + 1) * 20)) for client in range(4)}, model)
        with contextlib.redirect_stdout(io.StringIO()):
            update = agents[0].local_train(model, torch.nn.CrossEntropyLoss(), 1, neurotoxin_mask={})
        self.assertLessEqual(float(update.norm()), float(before.norm()) * .10001)
        self.assertGreater(float(update.norm()), float(before.norm()) * .099)
        self.assertIs(helpers.core.utils.vector_to_model_wo_load, original_utility)
        self.assertIs(helpers.Agent.local_train, original_train)

    def test_native_agents_two_rounds_all_available_methods(self):
        from . import runner
        helpers = runner.native_helpers()
        hashes = []
        combinations = [(method, attack) for method in ["fedavg", "rlr", "flame", "lockdown", "crfl", "prodigal"]
                        for attack in (["badnet", "DBA"] if method == "lockdown" else ["badnet", "DBA", "neurotoxin", "pgd"])]
        for method, attack in combinations:
            with self.subTest(method=method, attack=attack), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / method
                argv = ["--method", method, "--attack", attack, "--output", str(output), "--rounds", "2", "--num-agents", "4",
                        "--num-corrupt", "1", "--bs", "2", "--local-ep", "1", "--max-batches", "1",
                        "--test-limit", "20", "--rlr-threshold", "2", "--crfl-clip", "10",
                        "--crfl-noise", ".001", "--crfl-test-noise", ".001", "--crfl-samples", "2"]
                settings = parse_args(argv)
                original_cwd = Path.cwd()
                with patch.object(helpers.core.utils, "get_datasets", side_effect=lambda name: (FakeCifar(), FakeCifar(20))), \
                        patch.object(helpers.core.utils, "distribute_data_dirichlet", return_value={client: list(range(client * 20, (client + 1) * 20)) for client in range(4)}), \
                        patch.object(helpers.core.models, "get_model", side_effect=lambda *args: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 10))), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    runner.train(settings)
                metadata = json.loads((output / "config.json").read_text())
                self.assertEqual(metadata["status"], "completed")
                self.assertEqual(metadata["observed_rounds"], 2)
                self.assertTrue(metadata["smoke_only"])
                self.assertEqual(Path.cwd(), original_cwd)
                hashes.append((metadata["partition_sha256"], metadata["initial_model_sha256"]))
                self.assertEqual(len((output / "rounds.jsonl").read_text().splitlines()), 2)
                with self.assertRaises(FileExistsError):
                    runner.train(settings)
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
