
class AppStateTutorial:

    def _update_help_text(self):
        if self._sandbox_state == SandboxState.CONTROLLING_AGENT:
            assert False
            controls_str = self._get_controls_text()
            if len(controls_str) > 0:
                self._text_drawer.add_text(
                    controls_str, TextOnScreenAlignment.TOP_LEFT
                )

            status_str = self._get_status_text()
            if len(status_str) > 0:
                self._text_drawer.add_text(
                    status_str,
                    TextOnScreenAlignment.TOP_CENTER,
                    text_delta_x=-280,
                    text_delta_y=-50,
                )

            progress_str = f"{self._num_iter_episodes - (self._num_episodes_done + 1)} episodes remaining"
            self._text_drawer.add_text(
                progress_str,
                TextOnScreenAlignment.TOP_RIGHT,
                text_delta_x=370,
            )

        elif self._sandbox_state == SandboxState.TUTORIAL:
            controls_str = self._tutorial.get_help_text()
            if len(controls_str) > 0:
                self._text_drawer.add_text(
                    controls_str, TextOnScreenAlignment.TOP_LEFT
                )

            tutorial_str = self._tutorial.get_display_text()
            if len(tutorial_str) > 0:
                self._text_drawer.add_text(
                    tutorial_str,
                    TextOnScreenAlignment.TOP_CENTER,
                    text_delta_x=-280,
                    text_delta_y=-50,
                )        

    def _sim_update_tutorial(self, dt: float):
        # todo: get rid of this
        # Keyframes are saved by RearrangeSim when stepping the environment.
        # Because the environment is not stepped in the tutorial, we need to save keyframes manually for replay rendering to work.
        self.get_sim().gfx_replay_manager.save_keyframe()

        self._tutorial.update(dt)

        if self.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
            self._tutorial.skip_stage()

        if self._tutorial.is_completed():
            self._tutorial.stop_animations()
            self._sandbox_state = SandboxState.CONTROLLING_AGENT
        else:
            self.cam_transform = self._tutorial.get_look_at_matrix()                