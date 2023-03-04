# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 02-12-2023
# =============================================================================
"""
This module contains an agent that forward and receive messages from a remote 
  agent through websocket. The remote agent should be a websocket server that
  accepts and returns json-formatted messages.
"""

from typing import Dict

import json
import logging

import websocket

from parlai.core.agents import Agent


def ws_start(host: str, port: int) -> websocket.WebSocket:
    """Connect to a websocket server and return the websocket object."""

    addr = f"ws://{host}:{port}/websocket"
    ws = websocket.WebSocket()

    try:
        ws.connect(addr)
        print(f"Connected to chatbot at < {addr} >.")

    except ConnectionRefusedError:
        print(f"Failed to connect to chatbot at < {addr} >. Maybe wrong host or port?")
        exit(1)

    return ws


def ws_send(ws: websocket.WebSocket, text: str) -> None:
    """Serialize dictionary and send data to websocket server.
        Also, save the data to chat history."""

    err_msg = "Websocket server is not connected."
    
    if ws is None:
        raise RuntimeError(err_msg)

    if ws.connected:
        data = {"role": "User", "text": text}
        ws.send(json.dumps(data))


def ws_recv(ws: websocket.WebSocket) -> Dict:
    """Receive data from websocket server and deserialize it. 
        Also, save the data to chat history."""
    
    err_msg = "Websocket server is not connected."
    
    if ws is None:
        raise RuntimeError(err_msg)
    
    if ws.connected:
        data = json.loads(ws.recv())
        data.update({"role": "Bot"})

        return data


class RemoteAgent(Agent):
    """Agent that forward and receive messages from a remote agent through
        websocket. The remote agent should be a websocket server that accepts
        and returns json-formatted messages."""

    def __init__(self, opt, shared=None):
        super().__init__(opt)

        logging.debug("CONFIG:\n" + json.dumps(opt, indent=True))

        self.id = __class__.__name__

        self.host_bot = opt["host_bot"]
        self.port_bot = opt["port_bot"]
        self.ws = ws_start(self.host_bot, self.port_bot)

    def act(self) -> Dict:
        """Send a message to the remote agent and receive a response."""

        observation = self.observation
        if observation is None:
            return {"id": self.getID(),
                    "text": "Nothing to reply to yet",
                    "episode_done": False}

        ws_send(self.ws, observation["text"])
        response = ws_recv(self.ws)
        response.update({"id": self.getID(), 
                         "episode_done": False})
        
        return response 
