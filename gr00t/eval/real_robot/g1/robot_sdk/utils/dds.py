from threading import Lock
from typing import Optional

import logging_mp
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


logger_mp = logging_mp.get_logger(__name__)

_dds_init_lock = Lock()
_dds_initialized = False
_dds_domain_id: Optional[int] = None
_dds_network_interface: Optional[str] = None


def ensure_channel_factory_initialized(domain_id: int, network_interface: Optional[str] = None) -> bool:
    """
    Initialize Unitree DDS channel factory once per process.

    Returns True if this call performed initialization, False if it was already initialized.
    """
    global _dds_initialized, _dds_domain_id, _dds_network_interface

    with _dds_init_lock:
        if _dds_initialized:
            if domain_id != _dds_domain_id:
                logger_mp.warning(
                    "DDS already initialized on domain %s; requested domain %s. Reusing existing initialization.",
                    _dds_domain_id,
                    domain_id,
                )
            return False

        try:
            if network_interface:
                ChannelFactoryInitialize(domain_id, network_interface)
                _dds_network_interface = network_interface
            else:
                ChannelFactoryInitialize(domain_id)
                _dds_network_interface = None
        except Exception:
            if network_interface:
                logger_mp.warning(
                    "DDS init failed on interface '%s'; retrying with autodetected interface.",
                    network_interface,
                )
                ChannelFactoryInitialize(domain_id)
                _dds_network_interface = None
            else:
                raise

        _dds_initialized = True
        _dds_domain_id = domain_id
        logger_mp.info(
            "DDS initialized (domain=%s, interface=%s).",
            _dds_domain_id,
            _dds_network_interface if _dds_network_interface else "autodetect",
        )
        return True
