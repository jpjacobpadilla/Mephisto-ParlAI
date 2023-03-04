# -*- coding: utf-8 -*-
# =============================================================================
# Author: Yuxuan Wang
# Date: 12-06-2022
# =============================================================================
"""
This module contains a dummy agent that simply echos everything it observed.
"""

from typing import Dict

import json
import logging

from parlai.core.agents import Agent


class EchoAgent(Agent):
    """Dummy agent that simply echos everything it observed. On initialization,
        it also print out all options it receives."""

    def __init__(self, opt, shared=None):
        super().__init__(opt)

        logging.debug("CONFIG:\n" + json.dumps(opt, indent=True))

        self.id = __class__.__name__
        if shared is None:
            logging.debug("CREATED FROM PROTOTYPE")
        else:
            logging.debug("CREATED FROM COPY")

    def share(self) -> Dict:
        """Copy response function <self.resp_fn>"""

        logging.debug("MODEL COPIED")

        shared = super().share()
        return shared

    def act(self) -> Dict:
        """Simply copy the input messages"""

        observation = self.observation
        if observation is None:
            return {"text": "Nothing to reply to yet"}

        response = observation.get("text", "I don\"t know")
        return {"id": self.getID(),
                "text": f"[ echo ] :: {response}"}
