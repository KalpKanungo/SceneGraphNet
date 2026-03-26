def get_relation(subj_box, obj_box):
    sx, sy, sw, sh = subj_box
    ox, oy, ow, oh = obj_box

    # centers
    scx, scy = sx + sw / 2, sy + sh / 2
    ocx, ocy = ox + ow / 2, oy + oh / 2

    # vertical relation
    if scy < ocy - 20:
        return "above"
    elif scy > ocy + 20:
        return "below"

    # horizontal relation
    if scx < ocx - 20:
        return "left_of"
    elif scx > ocx + 20:
        return "right_of"

    return "near"