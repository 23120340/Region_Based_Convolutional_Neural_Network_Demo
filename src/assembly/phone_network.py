"""Find the address a phone on the same LAN can use to reach this PC.

Nothing here is hard-coded: candidates come from the machine's network
interfaces at the moment a QR code is generated, ranked so the adapter that
carries the default route on a private network comes first.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit


# Adapter names that are almost never reachable from a phone on Wi-Fi.
_VIRTUAL_HINTS = (
    "vethernet", "virtualbox", "vmware", "vmnet", "docker", "br-", "veth",
    "wsl", "hyper-v", "loopback", "vboxnet", "npcap", "utun", "zerotier",
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


@dataclass(frozen=True)
class LanAddress:
    ip: str
    interface: str = ""
    is_default_route: bool = False

    @property
    def is_private_lan(self) -> bool:
        try:
            address = ipaddress.ip_address(self.ip)
        except ValueError:  # hostname given with --phone-host
            return True
        return address.is_private and not address.is_loopback and not address.is_link_local

    @property
    def is_virtual(self) -> bool:
        name = self.interface.casefold()
        return any(hint in name for hint in _VIRTUAL_HINTS)


def _default_route_ip() -> str | None:
    """IP of the adapter used for outbound traffic; no packet is sent."""
    for probe in ("192.0.2.1", "10.255.255.255"):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                sock.connect((probe, 9))
                return sock.getsockname()[0]
            except OSError:
                continue
    return None


def _interface_addresses() -> list[tuple[str, str]]:
    try:
        import psutil  # installed with ultralytics
    except ImportError:
        psutil = None
    pairs: list[tuple[str, str]] = []
    if psutil is not None:
        stats = psutil.net_if_stats()
        for name, addresses in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            pairs.extend((name, item.address) for item in addresses if item.family == socket.AF_INET)
        return pairs
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        ips = []
    return [("", ip) for ip in ips]


def rank_addresses(pairs: list[tuple[str, str]], default_ip: str | None) -> list[LanAddress]:
    """Keep IPv4 addresses a phone could reach and sort the best first."""
    found: dict[str, LanAddress] = {}
    for name, ip in pairs + ([("", default_ip)] if default_ip else []):
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if address.version != 4 or address.is_loopback or address.is_link_local or address.is_unspecified:
            continue
        existing = found.get(ip)
        found[ip] = LanAddress(
            ip=ip,
            interface=name or (existing.interface if existing else ""),
            is_default_route=ip == default_ip,
        )

    def score(item: LanAddress) -> tuple:
        address = ipaddress.ip_address(item.ip)
        return (
            not item.is_private_lan,
            item.is_virtual,
            not item.is_default_route,
            address in _CGNAT,
            # Home/office Wi-Fi ranges are the most common phone networks.
            not item.ip.startswith("192.168."),
            item.ip,
        )

    return sorted(found.values(), key=score)


def discover_lan_addresses() -> list[LanAddress]:
    return rank_addresses(_interface_addresses(), _default_route_ip())


def normalize_public_url(value: str) -> str:
    """Validate a tunnel/reverse-proxy base URL such as https://x.trycloudflare.com."""
    parts = urlsplit(value.strip())
    if parts.scheme not in {"https", "http"} or not parts.netloc:
        raise ValueError("--phone-public-url phải có dạng https://host[:port]")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
