import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from zeus.monitor.energy import ZeusMonitor
from zeus.monitor.power import PowerMonitor

from prometheus_client import Gauge, start_http_server


class BaseExperiment:
    """The basement class include initialize all techniques"""
    def __init__(self, energy_gauge, power_gauge,
                 batch_size=32, use_amp=False, early_stopping=False, max_epochs=10):
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs

        self.energy_gauge = energy_gauge
        self.power_gauge  = power_gauge
        # Data và model chung
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=True)

        self.model = models.resnet18(num_classes=10).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Monitors
        self.power_monitor = PowerMonitor()
        self.zeus_monitor  = ZeusMonitor()

        # AMP scaler nếu cần
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def train_one_epoch(self, epoch):
        self.model.train()
        start = time.time()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        # đo power sau mỗi epoch
        latest = self.power_monitor.get_power() or {}
        samples = list(latest.values())
        avg_power = sum(samples) / len(samples) if samples else 0.0
        print(f"Epoch {epoch} done in {time.time() - start:.1f}s | Power ≈ {avg_power:.1f} W")
        self.power_gauge.set(avg_power)

    def run(self):
        # Prepare a small validation set
        val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        val_loader = DataLoader(val_ds, batch_size=256)

        self.zeus_monitor.begin_window("full_training")
        best_val_loss = float('inf')
        patience = 3
        wait = 0

        for epoch in range(self.max_epochs):
            self.train_one_epoch(epoch)

            # Compute validation loss
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.cuda(), y.cuda()
                    logits = self.model(x)
                    val_loss += self.criterion(logits, y).item() * x.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

            # Early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Stopping early at epoch {epoch} (no val loss improvement for {patience} epochs)")
                    break

        measurement = self.zeus_monitor.end_window("full_training")
        energy = measurement.total_energy
        print(f"Total energy (J): {energy:.2f}")
        self.energy_gauge.set(energy)

class BaselineExperiment(BaseExperiment):
    def __init__(self, energy_gauge, power_gauge):
        super().__init__(
            energy_gauge=energy_gauge,
            power_gauge=power_gauge,
            batch_size=32,
            use_amp=False,
            early_stopping=False,
        )

class MixedPrecisionExperiment(BaseExperiment):
    def __init__(self, energy_gauge, power_gauge):
        super().__init__(
            energy_gauge=energy_gauge,
            power_gauge=power_gauge,
            batch_size=32,
            use_amp=True,
            early_stopping=False,
        )

class LargerBatchExperiment(BaseExperiment):
    def __init__(self, energy_gauge, power_gauge):
        super().__init__(
            energy_gauge=energy_gauge,
            power_gauge=power_gauge,
            batch_size=128,
            use_amp=False,
            early_stopping=False,
        )

class EarlyStoppingExperiment(BaseExperiment):
    def __init__(self, energy_gauge, power_gauge):
        super().__init__(
            energy_gauge=energy_gauge,
            power_gauge=power_gauge,
            batch_size=32,
            use_amp=False,
            early_stopping=True,
        )

def main():
    # Initialize Prometheus server
    start_http_server(8000)
    # Prometheus gauges
    energy_gauge = Gauge('ml_total_energy_joule', 'Total energy used by ML training')
    power_gauge  = Gauge('ml_current_power_watt',   'Most recent GPU power draw (W)')
    # Iterate experiment
    for exp_cls in [EarlyStoppingExperiment]:
        print(f"\n--- Running {exp_cls.__name__} ---")
        exp = exp_cls(energy_gauge=energy_gauge, power_gauge=power_gauge)
        exp.run()

if __name__ == "__main__":
    main()


