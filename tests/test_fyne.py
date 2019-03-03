from fyne import blackscholes


def test_impliedvol():
    vol, k, t = 0.2, 0.1, 0.5
    c = blackscholes.formula(k, t, vol)
    iv = blackscholes.implied_vol(k, t, c)

    assert abs(vol - iv) < 1e-6
