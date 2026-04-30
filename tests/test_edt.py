import numpy as np
import pytest

from persviz.edt import signed_edt


def test_signed_edt_zero_on_boundary():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[3:7, 3:7] = 1
    phi = signed_edt(img)
    assert phi[3, 3] == pytest.approx(-1.0)
    assert phi[3, 4] == pytest.approx(-1.0)
    assert phi[2, 5] == pytest.approx(1.0)


def test_signed_edt_sign_convention():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[8:12, 8:12] = 1
    phi = signed_edt(img)
    assert phi[10, 10] < 0
    assert phi[0, 0] > 0


def test_signed_edt_disk_radius():
    N = 80
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(yy - N / 2, xx - N / 2)
    disk = (r <= 15).astype(np.uint8)
    phi = signed_edt(disk)
    assert phi[N // 2, N // 2] == pytest.approx(-15.0, abs=1.5)
