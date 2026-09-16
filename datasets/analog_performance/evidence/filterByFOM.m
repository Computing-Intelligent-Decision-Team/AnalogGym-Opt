function filteredTable = filterByFOM(table)
    % 获取 all_combined_data{1} 表格的3到17列
    subsetData = table(:, 3:17);

    % 使用 unique 函数在这个子集上找出唯一的行，同时保留原始顺序
    [~, idx] = unique(subsetData, 'rows', 'stable');

    % 使用这些索引来获取原始表格中的唯一行
    uniqueData = table(idx, :);
    disp(['Unique rows after initial filtering: ', num2str(height(uniqueData))]);

    % 过滤表格中 FOMS 和 FOML 小于 10 的行
    fomsIndex = uniqueData.FOMS >= 10;
    fomlIndex = uniqueData.FOML >= 10;
    areaIndex = uniqueData.chip_area <= 500;
    combinedIndex = fomsIndex & fomlIndex & areaIndex; % 两条件都满足
    filteredTable = uniqueData(combinedIndex, :);
    disp(['Rows after filtering area, FOMS and FOML: ', num2str(height(filteredTable))]);

    % 删除 d_settle 和 settlingTime 项相同的行
    % 找出 d_settle 和 settlingTime 相等的行的索引
    unequalIndex = filteredTable.d_settle ~= filteredTable.settlingTime;

    % 仅保留 d_settle 和 settlingTime 不相等的行
    filteredTable = filteredTable(unequalIndex, :);
    disp(['Rows after removing d_settle = settlingTime: ', num2str(height(filteredTable))]);
end
