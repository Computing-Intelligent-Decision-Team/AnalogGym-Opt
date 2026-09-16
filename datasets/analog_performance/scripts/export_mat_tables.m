% Read-only extraction of MATLAB tables to plain numeric MAT intermediates.
sourceRoot = getenv('ANALOG_SOURCE');
outRoot = getenv('ANALOG_OUTPUT');
assert(isfolder(sourceRoot) && isfolder(outRoot));
files = dir(fullfile(sourceRoot, '**', 'Perf_pop_data.mat'));
[~, order] = sort(string(fullfile({files.folder}, {files.name})));
files = files(order);
manifest = struct([]);
for k = 1:numel(files)
    sourcePath = fullfile(files(k).folder, files(k).name);
    loaded = load(sourcePath);
    names = fieldnames(loaded);
    tableNames = names(cellfun(@(n) istable(loaded.(n)), names));
    assert(numel(tableNames) == 1, 'Expected exactly one table: %s', sourcePath);
    T = loaded.(tableNames{1});
    assert(all(varfun(@(x) isnumeric(x) && isreal(x) && size(x,2)==1, T, 'OutputFormat','uniform')), 'Non-scalar numeric column');
    values = table2array(T);
    columns = T.Properties.VariableNames;
    units = T.Properties.VariableUnits;
    descriptions = T.Properties.VariableDescriptions;
    rowNames = T.Properties.RowNames;
    relativePath = strrep(sourcePath(numel(sourceRoot)+2:end), '\', '/');
    destination = fullfile(outRoot, '_work', sprintf('source_%03d.mat', k));
    save(destination, 'values', 'columns', 'units', 'descriptions', 'rowNames', 'relativePath', '-v7');
    entry = struct('source_relative_path', relativePath, 'intermediate', sprintf('source_%03d.mat', k), ...
        'variable', tableNames{1}, 'rows', height(T), 'columns', {columns}, 'units', {units}, ...
        'descriptions', {descriptions}, 'source_bytes', files(k).bytes);
    if isempty(manifest), manifest = entry; else, manifest(end+1) = entry; end
    fprintf('%d/%d rows=%d %s\n', k, numel(files), height(T), relativePath);
end
fid=fopen(fullfile(outRoot, '_work', 'export_manifest.json'),'w','n','UTF-8');
assert(fid>0);
fprintf(fid,'%s',jsonencode(manifest));
fclose(fid);
fprintf('Exported %d files, %d rows\n', numel(manifest), sum([manifest.rows]));
