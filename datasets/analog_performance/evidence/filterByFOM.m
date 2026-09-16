function filteredTable = filterByFOM(table)
    % Select columns 3 through 17 from the input table.
    subsetData = table(:, 3:17);

    % Find unique rows in this subset, preserving their original order.
    [~, idx] = unique(subsetData, 'rows', 'stable');

    % Use these indices to select the corresponding full rows.
    uniqueData = table(idx, :);
    disp(['Unique rows after initial filtering: ', num2str(height(uniqueData))]);

    % Exclude rows with FOMS or FOML below 10.
    fomsIndex = uniqueData.FOMS >= 10;
    fomlIndex = uniqueData.FOML >= 10;
    areaIndex = uniqueData.chip_area <= 500;
    combinedIndex = fomsIndex & fomlIndex & areaIndex; % Require all three conditions.
    filteredTable = uniqueData(combinedIndex, :);
    disp(['Rows after filtering area, FOMS and FOML: ', num2str(height(filteredTable))]);

    % Remove rows where d_settle equals settlingTime.
    % Find rows where d_settle differs from settlingTime.
    unequalIndex = filteredTable.d_settle ~= filteredTable.settlingTime;

    % Keep only rows where the settling fields differ.
    filteredTable = filteredTable(unequalIndex, :);
    disp(['Rows after removing d_settle = settlingTime: ', num2str(height(filteredTable))]);
end
