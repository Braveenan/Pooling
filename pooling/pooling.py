import torch

class Pooling:
    def __init__(self, pooling_type):
        if "-" in pooling_type:
            self.pooling_type, self.pooling_param = pooling_type.split("-")
        else:
            self.pooling_type = pooling_type
            self.pooling_param = None

        self.pooling_operations = {
            "mean": self.mean_pooling,
            "max": self.max_pooling,
            "mix": self.mixed_pooling,
            "lnp": self.learned_norm_pooling,
            "smp": self.softmax_pooling,
            "lse": self.log_sum_exp_pooling,
            "std": self.std_pooling,         # Standard deviation pooling
            "var": self.variance_pooling,     # Variance pooling
            "skew": self.skewness_pooling,   # Skewness pooling
            "kurt": self.kurtosis_pooling     # Kurtosis pooling
        }

        if self.pooling_type not in self.pooling_operations:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    def mean_pooling(self, x, d):
        return x.mean(dim=d)

    def max_pooling(self, x, d):
        return x.max(dim=d).values

    def mixed_pooling(self, x, d):
        max_pool = self.max_pooling(x, d)
        mean_pool = self.mean_pooling(x, d)
        mix_ratio = float(self.pooling_param) / 100.0
        return mix_ratio * max_pool + (1 - mix_ratio) * mean_pool
    
    def learned_norm_pooling(self, x, d):
        p = int(self.pooling_param)
        n = x.size(d)
        norm_pool = torch.pow(torch.sum(torch.abs(x) ** p, dim=d) / n, 1.0 / p)
        return norm_pool
    
    def softmax_pooling(self, x, d):
        lambda_param = int(self.pooling_param)
        softmax_weights = torch.softmax(lambda_param * x, dim=d)
        softmax_pool = torch.sum(softmax_weights * x, dim=d)
        return softmax_pool

    def log_sum_exp_pooling(self, x, d):
        r = float(self.pooling_param) / 100.0
        n = x.size(d)
        lse_pool = (1 / r) * torch.log(torch.sum(torch.exp(r * x), dim=d) / n)
        return lse_pool

    def std_pooling(self, x, d):
        mean = self.mean_pooling(x, d)
        std_dev = torch.sqrt(((x - mean.unsqueeze(d)) ** 2).mean(dim=d))
        return std_dev

    def variance_pooling(self, x, d):
        mean = self.mean_pooling(x, d)
        variance = ((x - mean.unsqueeze(d)) ** 2).mean(dim=d)
        return variance

    def skewness_pooling(self, x, d):
        mean = self.mean_pooling(x, d)
        std_dev = self.std_pooling(x, d)
        skewness = ((x - mean.unsqueeze(d)) ** 3).mean(dim=d) / (std_dev ** 3)
        return skewness

    def kurtosis_pooling(self, x, d):
        mean = self.mean_pooling(x, d)
        std_dev = self.std_pooling(x, d)
        kurtosis = ((x - mean.unsqueeze(d)) ** 4).mean(dim=d) / (std_dev ** 4)
        return kurtosis

    def get_vector_after_pooling(self, data, dim):
        return self.pooling_operations[self.pooling_type](data, dim)
