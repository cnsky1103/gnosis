import sys
import types

sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

import main as main_module


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1

    def kill(self):
        self.killed = True


def test_sleep_prevention_starts_caffeinate_on_macos(monkeypatch):
    calls = []
    fake_process = FakeProcess()

    monkeypatch.setattr(main_module.sys, "platform", "darwin")
    monkeypatch.setattr(main_module.shutil, "which", lambda name: "/usr/bin/caffeinate")
    monkeypatch.setattr(main_module.os, "getpid", lambda: 12345)

    def fake_popen(args, stdout=None, stderr=None):
        calls.append((args, stdout, stderr))
        return fake_process

    monkeypatch.setattr(main_module.subprocess, "Popen", fake_popen)

    with main_module.prevent_system_sleep_during_tts():
        assert calls == [
            (
                ["/usr/bin/caffeinate", "-ims", "-w", "12345"],
                main_module.subprocess.DEVNULL,
                main_module.subprocess.DEVNULL,
            )
        ]

    assert fake_process.terminated is True
    assert fake_process.killed is False
    assert fake_process.wait_calls == 1


def test_sleep_prevention_skips_non_macos(monkeypatch):
    calls = []

    monkeypatch.setattr(main_module.sys, "platform", "linux")
    monkeypatch.setattr(main_module.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))

    with main_module.prevent_system_sleep_during_tts():
        pass

    assert calls == []
