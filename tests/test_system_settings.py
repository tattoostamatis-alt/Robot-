"""Unit tests for system settings parsing (home_robot/system_settings.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.system_settings import (  # noqa: E402
    bt_args, parse_bt_devices, parse_devices, parse_volume, parse_wifi_list,
    quote_for_display, volume_args, wifi_connect_args,
)

# Real `nmcli -t -f IN-USE,SSID,SIGNAL,SECURITY device wifi list` shape.
WIFI = """\
*:HomeNet:78:WPA2
 :HomeNet:41:WPA2
 :Neighbour:55:WPA2
 :OpenCafe:30:
 ::22:WPA2
 :Weird\\:Name:60:WPA2
"""


# ── wifi ──────────────────────────────────────────────────────────────────────

def test_networks_are_parsed():
    nets = parse_wifi_list(WIFI)
    assert {n['ssid'] for n in nets} == {'HomeNet', 'Neighbour', 'OpenCafe',
                                         'Weird:Name'}


def test_strongest_first():
    sigs = [n['signal'] for n in parse_wifi_list(WIFI)]
    assert sigs == sorted(sigs, reverse=True)


def test_the_connected_network_is_flagged():
    active = [n for n in parse_wifi_list(WIFI) if n['active']]
    assert len(active) == 1 and active[0]['ssid'] == 'HomeNet'


def test_the_same_network_on_two_bands_appears_once():
    """Otherwise one network looks like two and picking the weak one fails."""
    nets = [n for n in parse_wifi_list(WIFI) if n['ssid'] == 'HomeNet']
    assert len(nets) == 1
    assert nets[0]['signal'] == 78          # the stronger sighting
    assert nets[0]['active']                # and it kept its active flag


def test_an_escaped_colon_in_an_ssid_survives():
    """‼️ nmcli -t escapes ':' inside fields. Splitting naively turns
    'Weird:Name' into an SSID of 'Weird' and a signal of 'Name'."""
    weird = [n for n in parse_wifi_list(WIFI) if n['ssid'] == 'Weird:Name']
    assert weird and weird[0]['signal'] == 60


def test_open_networks_are_marked_insecure():
    cafe = [n for n in parse_wifi_list(WIFI) if n['ssid'] == 'OpenCafe'][0]
    assert not cafe['secure']
    home = [n for n in parse_wifi_list(WIFI) if n['ssid'] == 'HomeNet'][0]
    assert home['secure']


def test_hidden_networks_are_dropped():
    """An unnamed row cannot be joined by name and only confuses the list."""
    assert all(n['ssid'] for n in parse_wifi_list(WIFI))


def test_empty_and_garbage_input():
    assert parse_wifi_list('') == []
    assert parse_wifi_list(None) == []
    assert parse_wifi_list('nonsense\n\n') == []


def test_non_numeric_signal_does_not_crash():
    assert parse_wifi_list(' :X:abc:WPA2')[0]['signal'] == 0


# ── interfaces ────────────────────────────────────────────────────────────────

DEVICES = """\
wlx00c0cab368ff:wifi:connected:HomeNet
tailscale0:tun:connected (externally):
eno1:ethernet:unavailable:
"""


def test_devices_are_parsed():
    devs = parse_devices(DEVICES)
    assert devs[0]['device'] == 'wlx00c0cab368ff'
    assert devs[0]['type'] == 'wifi'
    assert devs[0]['connection'] == 'HomeNet'


def test_devices_tolerate_a_missing_connection():
    assert parse_devices('eno1:ethernet:unavailable')[0]['connection'] == ''


def test_devices_empty():
    assert parse_devices('') == []


# ── bluetooth ─────────────────────────────────────────────────────────────────

BT = """\
Device AA:BB:CC:DD:EE:01 Living Room Speaker
Device AA:BB:CC:DD:EE:02 DualSense Wireless Controller
Device AA:BB:CC:DD:EE:03
"""


def test_bt_devices_are_parsed():
    devs = parse_bt_devices(BT)
    assert len(devs) == 3
    assert any(d['name'] == 'Living Room Speaker' for d in devs)


def test_a_nameless_device_shows_its_address():
    """Two empty rows cannot be told apart."""
    devs = parse_bt_devices(BT)
    nameless = [d for d in devs if d['mac'] == 'AA:BB:CC:DD:EE:03'][0]
    assert nameless['name'] == 'AA:BB:CC:DD:EE:03'


def test_connected_devices_come_first():
    devs = parse_bt_devices(BT, connected=['aa:bb:cc:dd:ee:02'])
    assert devs[0]['mac'] == 'AA:BB:CC:DD:EE:02'
    assert devs[0]['connected']


def test_connected_match_is_case_insensitive():
    devs = parse_bt_devices(BT, connected=['AA:BB:CC:DD:EE:01'])
    assert devs[0]['connected']


def test_bt_ignores_other_output():
    assert parse_bt_devices('Agent registered\nsomething else') == []


def test_bt_empty():
    assert parse_bt_devices('') == []
    assert parse_bt_devices(None) == []


# ── volume ────────────────────────────────────────────────────────────────────

def test_volume_is_parsed():
    out = 'Volume: front-left: 45875 /  70% / -9.29 dB,   front-right: 45875 /  70%'
    assert parse_volume(out) == 70


def test_volume_missing():
    assert parse_volume('') is None
    assert parse_volume(None) is None
    assert parse_volume('no percentage here') is None


# ── argument builders ─────────────────────────────────────────────────────────

def test_connect_args_without_a_password():
    assert wifi_connect_args('HomeNet') == [
        'nmcli', 'device', 'wifi', 'connect', 'HomeNet']


def test_connect_args_with_a_password():
    args = wifi_connect_args('HomeNet', 'hunter2')
    assert args[-2:] == ['password', 'hunter2']


def test_an_ssid_with_a_space_stays_one_argument():
    """‼️ Arguments are a LIST, never a shell string. An SSID arrives over the
    network and must never be concatenated into a command line."""
    args = wifi_connect_args('My Home; rm -rf /')
    assert args[4] == 'My Home; rm -rf /'
    assert len(args) == 5


def test_no_ssid_no_command():
    assert wifi_connect_args('') is None
    assert wifi_connect_args(None) is None


def test_bt_simple_actions():
    assert bt_args('on') == ['bluetoothctl', 'power', 'on']
    assert bt_args('devices') == ['bluetoothctl', 'devices']
    assert 'scan' in bt_args('scan')


def test_bt_device_actions_need_a_mac():
    assert bt_args('connect', 'AA:BB') == ['bluetoothctl', 'connect', 'AA:BB']
    assert bt_args('connect') is None


def test_bt_rejects_unknown_actions():
    assert bt_args('rm -rf /') is None
    assert bt_args('reboot', 'AA:BB') is None


def test_volume_is_clamped():
    """A slider that can send 400% is one bug away from a broken speaker."""
    assert volume_args(400)[-1] == '150%'
    assert volume_args(-20)[-1] == '0%'
    assert volume_args(55)[-1] == '55%'


def test_volume_rejects_nonsense():
    assert volume_args('loud') is None
    assert volume_args(None) is None


def test_display_quoting_is_safe():
    shown = quote_for_display(wifi_connect_args('My Home', 'p w'))
    assert "'My Home'" in shown
