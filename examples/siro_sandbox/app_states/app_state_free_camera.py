

class AppStateFreeCamera:

    def __init__(self):

        self.lookat = None      


    def _free_camera_lookat_control(self):
        if self.lookat is None:
            # init lookat
            self.lookat = np.array(
                self.get_sim().sample_navigable_point()
            ) + np.array([0, 1, 0])
        else:
            # update lookat
            move_delta = 0.1
            move = np.zeros(3)
            if self.gui_input.get_key(GuiInput.KeyNS.W):
                move[0] -= move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.S):
                move[0] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.O):
                move[1] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.P):
                move[1] -= move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.J):
                move[2] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.L):
                move[2] -= move_delta

            # align move forward direction with lookat direction
            rotation_rad = -self.lookat_offset_yaw
            rot_matrix = np.array(
                [
                    [np.cos(rotation_rad), 0, np.sin(rotation_rad)],
                    [0, 1, 0],
                    [-np.sin(rotation_rad), 0, np.cos(rotation_rad)],
                ]
            )

            self.lookat += mn.Vector3(rot_matrix @ move)

        # highlight the lookat translation as a red circle
        self._debug_line_render.draw_circle(
            self.lookat, 0.03, mn.Color3(1, 0, 0)
        )