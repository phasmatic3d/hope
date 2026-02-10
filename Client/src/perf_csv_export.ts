export type PointCloudPerfSample = {
    sampleIndex: number;
    frameId: number;
    decodeMs: number;
    renderMs: number;
    capturedAtIso: string;
};

/**
 * Stores per-frame performance samples in memory.
 * CSV content is generated only when the user requests a download.
 */
export function createPerfCsvExporter(filePrefix: string) {
    const samples: PointCloudPerfSample[] = [];
    let nextSampleIndex = 0;

    /**
     * Adds one performance sample to the in-memory list.
     * Samples are kept in arrival order for deterministic CSV output.
     */
    function addSample(frameId: number, decodeMs: number, renderMs: number) {
        samples.push({
            sampleIndex: nextSampleIndex,
            frameId,
            decodeMs,
            renderMs,
            capturedAtIso: new Date().toISOString(),
        });

        nextSampleIndex += 1;
    }

    /**
     * Builds CSV text from the current sample list.
     * The header matches the previous CSV schema for compatibility.
     */
    function createCsvText() {
        const header = 'sample_index,frame_id,decode_ms_full,render_ms,captured_at_iso';
        const rows = samples.map((sample) => [
            sample.sampleIndex,
            sample.frameId,
            sample.decodeMs.toFixed(3),
            sample.renderMs.toFixed(3),
            sample.capturedAtIso,
        ].join(','));

        return `${header}\n${rows.join('\n')}`;
    }

    /**
     * Downloads all currently buffered samples as a CSV file.
     * File naming is timestamped so repeated downloads do not overwrite.
     */
    function downloadCsvToDisk() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `${filePrefix}-${timestamp}.csv`;
        const csvText = createCsvText();
        const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = fileName;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();

        URL.revokeObjectURL(url);
    }

    return {
        addSample,
        getCount: () => samples.length,
        downloadCsvToDisk,
    };
}
