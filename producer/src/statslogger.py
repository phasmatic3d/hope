
from rich.table import Table
from encoding import EncodingStats
from encoding import GeneralStats
def make_encoding_stats_table(stats: EncodingStats, section: str, show_headers = True) -> Table:
    title = f"==== {section} ===="
    table = Table(title=title,box=None, padding=(0,1), show_header = show_headers)

    table.add_row("Draco Encode", f"{stats.encode_ms:.2f} ms")
    table.add_row("Points", str(stats.pts))
    saved_pct = (stats.raw_bytes - stats.encoded_bytes) / stats.raw_bytes * 100 if stats.raw_bytes else 0
    table.add_row("Raw/Enc Bytes", f"{stats.raw_bytes}/{stats.encoded_bytes} ({saved_pct:.1f}%)")
    return table
def make_general_stats_table(stats: GeneralStats, section: str, show_headers = True) -> Table:
    title = f"==== {section} ===="
    table = Table(title = title, box=None, padding=(0,1), show_header = show_headers)

    table.add_row("Obj Detection", f"{stats.det_ms:.2f} ms")
    table.add_row("Frame Prep", f"{stats.frame_ms:.2f} ms")
    table.add_row("Cull Prep", f"{stats.cull_ms:.2f} ms")
    table.add_row("Point Prep", f"{stats.pc_ms:.2f} ms")
    table.add_row("Draco Prep", f"{stats.prep_ms:.2f} ms")
    table.add_row("True Encoding Time ", f"{stats.true_enc_ms:.2f} ms")
    return table

def make_total_time_table(total_time : float) -> Table:
    
    table = Table( box=None, padding=(0,1))

    table.add_row("Total process time: ", f"{total_time:.2f} ms")

    return table