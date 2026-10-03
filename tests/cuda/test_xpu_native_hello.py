"""K0's prebuilt hello extension on the B70: an add, a sub-group-16 butterfly and one bf16 DPAS, exact against torch."""

import os

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available() or DEVICE != "xpu":
    pytest.skip("needs an XPU", allow_module_level=True)

pytestmark = [pytest.mark.xpu_kernel("native"),
              pytest.mark.skipif(not os.environ.get("TF_XPU_EXT_DIR"), reason="no prebuilt extensions (TF_XPU_EXT_DIR)")]


@pytest.fixture(scope="module")
def hello():
    from tensorfold.xpu.build import load

    return load("tensorfold_xpu_hello_v1")


def test_add(DEV, hello):
    a = torch.randn(1000, device=DEV)
    b = torch.randn(1000, device=DEV)
    assert torch.equal(hello.add(a, b), a + b)


def test_sub_group_butterfly_sums_each_group_of_16(DEV, hello):
    x = torch.randint(-1000, 1000, (64 * 16,), device=DEV).float()        # integers: every order gives the same sum
    want = x.reshape(-1, 16).sum(1, keepdim=True).expand(-1, 16).reshape(-1)
    assert torch.equal(hello.sg_sum(x), want)


def test_one_dpas_matches_torch(DEV, hello):
    g = torch.Generator(device=DEV).manual_seed(5)
    a = torch.randint(-8, 8, (8, 16), generator=g, device=DEV).bfloat16()   # small integers: the products and sums are
    b = torch.randint(-8, 8, (16, 16), generator=g, device=DEV).bfloat16()  # exact in fp32, so DPAS must match exactly
    assert torch.equal(hello.dpas(a, b), a.float() @ b.float())
