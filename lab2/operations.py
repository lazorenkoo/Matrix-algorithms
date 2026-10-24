from dataclasses import dataclass


@dataclass
class Operations:
    multiplications: int = 0
    additions_subtractions: int = 0
    divisions: int = 0

    @property
    def total(self) -> int:
        return self.multiplications + self.additions_subtractions + self.divisions
